"""
Stage1 SFT (scheme C): instruction + Uni-Mol + single-token BOS projection (BosProjectionLayer).
Response is enriched_description from LMDB; LLM is frozen; train Uni-Mol + BosProjection.
Dataset: tmQM stage1 train/val LMDB paths.
Each LMDB record needs atoms, coordinates, smiles, enriched_description.

Usage: CUDA_VISIBLE_DEVICES=2,3 deepspeed --num_gpus=2 Stage1.py
"""
import os
import json
import pickle
import lmdb
import numpy as np
import wandb
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from safetensors.torch import save_file
from torch.utils.data import Dataset
import utils  # noqa: F401
from utils import (
    ATOM_DIM,
    MAX_SEQ_LENGTH,
    BridgeUnimolCollator,
    _atoms_coords_remove_h_center,
    read_lmdb,
    tokenize_generation_sample_object_ref,
    unwrap_hf_model,
)
from bos_unimol_qwen_model import BosUnimolQwenModel, RECIPE_STAGE1
from train_defaults import STAGE1_DEFAULTS

INSTRUCTION = "Give me an introduction to this transition metal complex:"


class TmQMIntroBridgeUnimolDataset(Dataset):
    """tmQM stage1 LMDB: atoms, coordinates, smiles, enriched_description; BOS slot for frozen-LLM Stage1."""

    def __init__(self, lmdb_paths, tokenizer, max_length=512, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for p in lmdb_paths:
            if not os.path.exists(p):
                continue
            raw = read_lmdb(p, max_samples=max_samples)
            for d in raw:
                if "atoms" not in d or "coordinates" not in d or "smiles" not in d or "enriched_description" not in d:
                    continue
                desc = d.get("enriched_description")
                if not desc or (isinstance(desc, str) and not desc.strip()):
                    continue
                self.samples.append(d)
        if not self.samples and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            raise RuntimeError(
                f"No valid samples (need atoms, coordinates, smiles, enriched_description): {lmdb_paths}."
            )
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[TmQMIntroBridgeUnimolDataset] {len(self.samples)} samples; response=enriched_description.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        smiles = data["smiles"]
        response = data["enriched_description"]
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
        ids = tokenize_generation_sample_object_ref(
            self.tokenizer,
            user_content,
            response,
            prefix_len_mode="prefix",
            tokenizer_full=dict(truncation=True, max_length=self.max_length),
            tokenizer_split_parts=dict(truncation=True, max_length=self.max_length),
        )
        return {**ids, "atoms": atoms, "coordinates": coords}


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
        model.tokenizer.save_pretrained(output_dir)

        bos_proj_state = {f"bos_projection.{k}": v.cpu() for k, v in model.bos_projection_layer.state_dict().items()}
        save_file(bos_proj_state, os.path.join(output_dir, "bos_projection.safetensors"))

        torch.save({"model": model.unimol.state_dict()}, os.path.join(output_dir, "unimol.pt"))
        with open(os.path.join(output_dir, "bridge_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "scheme": "C",
                "atom_dim": ATOM_DIM,
                "bos_only": True,
            }, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage1: intro + Uni-Mol scheme C (BOS projection only), frozen LLM")
    parser.add_argument(
        "--model_name",
        type=str,
        default=STAGE1_DEFAULTS["model_name"],
        help="HF causal LM directory",
    )
    parser.add_argument(
        "--unimol_ckpt",
        type=str,
        default=STAGE1_DEFAULTS["unimol_ckpt"],
        help="Uni-Mol pretrained weights (.pt or directory with model.safetensors)",
    )
    parser.add_argument(
        "--unimol_dict",
        type=str,
        default=STAGE1_DEFAULTS["unimol_dict"],
    )
    parser.add_argument(
        "--train_lmdb",
        nargs="+",
        default=STAGE1_DEFAULTS["train_lmdb"],
        help="Training LMDB paths (one or more)",
    )
    parser.add_argument(
        "--val_lmdb",
        type=str,
        default=STAGE1_DEFAULTS["val_lmdb"],
        help="Validation LMDB path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=STAGE1_DEFAULTS["output_dir"],
    )
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
        wandb.init(project="sft_intro_stage1_scheme_c")
        print("[Stage1] intro + Uni-Mol scheme C (BOS single-token projection); response=enriched_description; LLM frozen.")

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = BosUnimolQwenModel(
        args_main.model_name,
        args_main.unimol_dict,
        recipe=RECIPE_STAGE1,
        unimol_checkpoint=args_main.unimol_ckpt,
    )
    train_dataset = TmQMIntroBridgeUnimolDataset(
        args_main.train_lmdb, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, max_samples=None
    )
    val_dataset = TmQMIntroBridgeUnimolDataset(
        [args_main.val_lmdb], tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, max_samples=None
    )
    if local_rank == 0:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    data_collator = BridgeUnimolCollator(tokenizer)
    ds_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config.json")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        deepspeed=ds_config,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
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
