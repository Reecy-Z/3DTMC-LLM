"""
Continue from Stage1 (BOS-only): load Uni-Mol + BosProjection, unfreeze LLM with LoRA.
Mixed data: (1) LMDB with polished_description; (2) JSON Q&A (text only).

Usage: CUDA_VISIBLE_DEVICES=2,3 deepspeed --num_gpus=2 Stage2.py
"""
import os
import json
import pickle
import lmdb
import numpy as np
import wandb
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from safetensors.torch import save_file, load_file
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm

import utils  # noqa: F401
from utils import (
    ATOM_DIM,
    MAX_SEQ_LENGTH,
    _atoms_coords_remove_h_center,
    read_lmdb,
    unwrap_hf_model,
)
from bos_unimol_qwen_model import BosUnimolQwenModel, RECIPE_STAGE2
from train_defaults import STAGE2_DEFAULTS

INSTRUCTION = "Give me an introduction to this transition metal complex:"


# ---------- LMDB dataset: response = polished_description ----------
class TmQMPolishedLMDBDataset(Dataset):
    """LMDB：INSTRUCTION + SMILES + atoms + coordinates -> polished_description"""
    def __init__(self, lmdb_paths, tokenizer, max_length=512, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for p in lmdb_paths:
            if not os.path.exists(p):
                continue
            raw = read_lmdb(p, max_samples=max_samples)
            for d in raw:
                if "atoms" not in d or "coordinates" not in d or "smiles" not in d:
                    continue
                desc = d.get("polished_description") or d.get("enriched_description")
                if not desc or (isinstance(desc, str) and not desc.strip()):
                    continue
                self.samples.append(d)
        if not self.samples and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            raise RuntimeError(f"No valid samples (need atoms, coordinates, smiles, polished_description): {lmdb_paths}")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[TmQMPolishedLMDBDataset] {len(self.samples)} samples; response=polished_description")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        smiles = data["smiles"]
        response = data.get("polished_description") or data.get("enriched_description", "")
        if not isinstance(response, str):
            response = str(response) if response is not None else ""
        atoms = data["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(data["coordinates"], dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
        user_content = f"{INSTRUCTION} {smiles}"
        messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": response}]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_str = self.tokenizer.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(full_text, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_enc = self.tokenizer(prefix_str, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])
        # Keep prefix_len within encoded length
        prefix_len = min(prefix_len, len(enc["input_ids"]) - 1) if len(enc["input_ids"]) > 1 else 0
        response_ids = enc["input_ids"][prefix_len:]
        sep = "<|im_end|>"
        before_3d_str = prefix_str.split(sep, 1)[0]
        after_3d_str = sep + prefix_str.split(sep, 1)[1]
        before_3d_enc = self.tokenizer(before_3d_str, return_tensors=None, truncation=True, max_length=self.max_length)
        after_3d_enc = self.tokenizer(after_3d_str, return_tensors=None, truncation=True, max_length=self.max_length)
        return {
            "sample_type": "lmdb",
            "before_3d_ids": before_3d_enc["input_ids"],
            "after_3d_ids": after_3d_enc["input_ids"],
            "response_ids": response_ids,
            "atoms": atoms,
            "coordinates": coords,
        }


# ---------- JSON Q&A dataset (no 3D) ----------
class JsonQADataset(Dataset):
    """JSON: question -> answer; no 3D."""
    def __init__(self, json_paths, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for p in json_paths:
            if not os.path.isfile(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        q = item.get("question")
                        a = item.get("answer")
                        if q is not None and a is not None and str(q).strip() and str(a).strip():
                            self.samples.append({"question": str(q).strip(), "answer": str(a).strip()})
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[JsonQADataset] {len(self.samples)} text-only Q&A samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        question = item["question"]
        answer = item["answer"]
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_str = self.tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(full_text, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_enc = self.tokenizer(prefix_str, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])
        # Cap prefix_len so response_ids is non-empty
        prefix_len = min(prefix_len, len(enc["input_ids"]) - 1) if len(enc["input_ids"]) > 1 else 0
        response_ids = enc["input_ids"][prefix_len:]
        return {
            "sample_type": "text",
            "before_3d_ids": prefix_enc["input_ids"][:prefix_len] if prefix_len > 0 else prefix_enc["input_ids"],
            "after_3d_ids": [],
            "response_ids": response_ids,
            "atoms": [],
            "coordinates": [],
        }


def _pad_ids(id_lists, pad_id, dtype=torch.long):
    B = len(id_lists) if id_lists else 0
    max_len = max(len(ids) for ids in id_lists) if id_lists else 0
    if max_len == 0:
        return torch.zeros(B, 0, dtype=dtype), torch.zeros(B, 0, dtype=torch.long)
    B = len(id_lists)
    padded = torch.full((B, max_len), pad_id, dtype=dtype)
    mask = torch.zeros(B, max_len, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        L = len(ids)
        padded[i, :L] = torch.tensor(ids, dtype=dtype)
        mask[i, :L] = 1
    return padded, mask


class MixedCollator:
    """Collate mixed batches: LMDB (3D) + JSON (text-only)."""
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        before_3d_ids = [b["before_3d_ids"] for b in batch]
        after_3d_ids = [b["after_3d_ids"] if b["after_3d_ids"] else [] for b in batch]
        response_ids = [b["response_ids"] for b in batch]
        list_atoms = [b["atoms"] for b in batch]
        list_coordinates = [b["coordinates"] for b in batch]
        sample_types = [b["sample_type"] for b in batch]
        before_3d_padded, before_3d_mask = _pad_ids(before_3d_ids, self.pad_id)
        after_3d_padded, after_3d_mask = _pad_ids(after_3d_ids, self.pad_id)
        response_padded, response_mask = _pad_ids(response_ids, self.pad_id)
        return {
            "before_3d_ids": before_3d_padded,
            "before_3d_mask": before_3d_mask,
            "after_3d_ids": after_3d_padded,
            "after_3d_mask": after_3d_mask,
            "response_ids": response_padded,
            "response_mask": response_mask,
            "list_atoms": list_atoms,
            "list_coordinates": list_coordinates,
            "sample_types": sample_types,
            "labels": response_padded,
        }


class BridgeUnimolTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)
        model = unwrap_hf_model(self.model)
        model.llm.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)

        bos_proj_state = {f"bos_projection.{k}": v.cpu() for k, v in model.bos_projection_layer.state_dict().items()}
        save_file(bos_proj_state, os.path.join(output_dir, "bos_projection.safetensors"))

        torch.save({"model": model.unimol.state_dict()}, os.path.join(output_dir, "unimol.pt"))
        with open(os.path.join(output_dir, "bridge_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_bridge_tokens": 0,
                    "atom_dim": ATOM_DIM,
                    "bos_only": True,
                    "use_bridge": False,
                },
                f,
                indent=2,
            )

    def _load_best_model(self):
        ckpt_dir = self.state.best_model_checkpoint
        if ckpt_dir is None:
            return
        model = unwrap_hf_model(self.model)
        model.llm.load_adapter(ckpt_dir, "default")

        bos_proj_path = os.path.join(ckpt_dir, "bos_projection.safetensors")
        if os.path.isfile(bos_proj_path):
            st = load_file(bos_proj_path, device="cpu")
            bos_state = {k.replace("bos_projection.", ""): v for k, v in st.items() if k.startswith("bos_projection.")}
            if bos_state:
                model.bos_projection_layer.load_state_dict(bos_state, strict=False)

        unimol_pt = os.path.join(ckpt_dir, "unimol.pt")
        if os.path.isfile(unimol_pt):
            state = torch.load(unimol_pt, map_location="cpu", weights_only=False)
            if "model" in state:
                state = state["model"]
            model.unimol.load_state_dict(state, strict=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage2: Uni-Mol BOS-only + LoRA, mixed LMDB + JSON QA")
    parser.add_argument(
        "--model_name",
        type=str,
        default=STAGE2_DEFAULTS["model_name"],
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=STAGE2_DEFAULTS["adapter"],
        help="Stage1 BOS-only checkpoint directory",
    )
    parser.add_argument(
        "--unimol_dict",
        type=str,
        default=STAGE2_DEFAULTS["unimol_dict"],
    )
    parser.add_argument(
        "--train_lmdb",
        nargs="+",
        default=STAGE2_DEFAULTS["train_lmdb"],
    )
    parser.add_argument(
        "--val_lmdb",
        type=str,
        default=STAGE2_DEFAULTS["val_lmdb"],
    )
    parser.add_argument(
        "--json_qa",
        nargs="+",
        default=STAGE2_DEFAULTS["json_qa"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=STAGE2_DEFAULTS["output_dir"],
    )
    # LoRA hyperparameters (same style as SFT_dipole_bridge_unimol_full.py)
    parser.add_argument("--lora_r", type=int, default=STAGE2_DEFAULTS["lora_r"], help="LoRA rank (default 8)")
    parser.add_argument("--lora_alpha", type=int, default=STAGE2_DEFAULTS["lora_alpha"], help="LoRA alpha (default 32)")
    parser.add_argument("--lora_target", type=str, default=STAGE2_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"],
                        help="LoRA target modules: qv=[q,v]_proj, qkv=[q,k,v]_proj, all=[q,k,v,o,gate,up,down]_proj")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=STAGE2_DEFAULTS["epochs"], help="Number of epochs")
    parser.add_argument("--lr", type=float, default=STAGE2_DEFAULTS["lr"], help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=STAGE2_DEFAULTS["batch_size"], help="per_device_train_batch_size")
    parser.add_argument("--local_rank", type=int, default=-1)
    args_main = parser.parse_args()
    try:
        import multiprocessing
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    output_dir = args_main.output_dir

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.init(project="sft_intro_bridge_unimol_continue_bos_only")
        print(f"[Continue] Mixed LMDB (polished_description) + JSON QA; LLM unfrozen (LoRA); 3D=BOS-only")
        print(f"[Continue] LoRA: r={args_main.lora_r}, alpha={args_main.lora_alpha}, target={args_main.lora_target}")

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = BosUnimolQwenModel(
        args_main.model_name,
        args_main.unimol_dict,
        recipe=RECIPE_STAGE2,
        adapter_path=args_main.adapter,
        lora_r=args_main.lora_r,
        lora_alpha=args_main.lora_alpha,
        lora_target=args_main.lora_target,
    )

    lmdb_dataset = TmQMPolishedLMDBDataset(
        args_main.train_lmdb, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH
    )
    json_dataset = JsonQADataset(args_main.json_qa, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH)
    train_dataset = ConcatDataset([lmdb_dataset, json_dataset])

    val_dataset = TmQMPolishedLMDBDataset(
        [args_main.val_lmdb], tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH
    )

    if local_rank == 0:
        print(f"Train: LMDB {len(lmdb_dataset)} + JSON {len(json_dataset)} = {len(train_dataset)}")
        print(f"Val: {len(val_dataset)} (LMDB test set)")

    data_collator = MixedCollator(tokenizer)
    ds_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config.json")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args_main.epochs,
        deepspeed=ds_config,
        per_device_train_batch_size=args_main.batch_size,
        per_device_eval_batch_size=args_main.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args_main.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.05,
        logging_steps=10,
        save_steps=1000,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=1000,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        save_total_limit=5,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = BridgeUnimolTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)
    trainer.train()
