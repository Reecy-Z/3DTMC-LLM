"""
Stage2 training: default mixed SFT (enriched_description + JSON QA) or description ablations.

Default (Stage2 from Stage1):
  CUDA_VISIBLE_DEVICES=2,3 deepspeed --num_gpus=2 Stage2.py

Description ablations (--ablation):
  freeze_3d   Frozen 3D encoder + projection; train LoRA only; polished_description + JSON QA
  random_3d   Random structure-slot embedding; train LoRA only
  multi_token Learnable multi-token query projection; instruction + SMILES; polished_description
  3d_only     3D slot only (no SMILES in prompt); polished_description

Examples:
  deepspeed --num_gpus=2 Stage2.py --ablation freeze_3d --3D_encoder_ckpt /path/to/encoder
  deepspeed --num_gpus=2 Stage2.py --ablation multi_token --train_lmdb /path/to/train.lmdb
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json

import numpy as np
import torch
import wandb
from safetensors.torch import load_file, save_file
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
import utils  # noqa: F401
from utils import (
    ATOM_DIM,
    DESCRIPTION_SAVE_KWARGS,
    DescriptionTrainer,
    InferenceOnlyCheckpointMixin,
    MAX_SEQ_LENGTH,
    MixedCollator,
    MultimodalCollator,
    MultimodalFullTrainer,
    SINGLE_TOKEN_PROJECTION_SAFETENSORS,
    THREE_D_ENCODER_STATE_PT,
    _atoms_coords_remove_h_center,
    read_lmdb,
    require_existing_model_path,
    resolve_three_d_encoder_ckpt,
    strip_single_token_projection_state_dict,
    unwrap_hf_model,
)
from multimodal_LLM import (
    RECIPE_STAGE2,
    RECIPE_STAGE3,
    MultimodalModel,
    MultimodalModelFreeze3D,
    MultimodalModelMultiToken,
    MultimodalModelRandom3D,
    MultimodalMultiTokenTrainer,
)
from train_defaults import DESCRIPTION_DEFAULTS, PROPERTY_DEFAULTS, STAGE2_DEFAULTS
from task_datasets import (
    RESPONSE_KEY,
    JsonQADataset,
    TmQMEnrichedLMDBDataset,
    TmQMEnrichedTokenizedDataset,
)

ABLATION_CHOICES = ("stage2", "freeze_3d", "random_3d", "multi_token", "3d_only")

_WANDB_PROJECT = {
    "stage2": "Stage2",
    "freeze_3d": "Description_ablation_freeze_3d",
    "random_3d": "Description_ablation_random_3d",
    "multi_token": "Description_ablation_multi_token",
    "3d_only": "Description_ablation_3d_only",
}

_DEFAULT_OUTPUT = {
    "stage2": STAGE2_DEFAULTS["output_dir"],
    "freeze_3d": "Description_freeze_3d_ckpt",
    "random_3d": "Description_random_3d_ckpt",
    "multi_token": "Description_multi_token_ckpt",
    "3d_only": "Description_3d_only_ckpt",
}


# ---------- Stage2 default dataset (enriched_description) ----------
INSTRUCTION_STAGE2 = "Give me a description of this transition metal complex:"


class Stage2EnrichedLMDBDataset(Dataset):
    """LMDB: INSTRUCTION + SMILES + atoms + coordinates -> enriched_description."""

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
                f"No valid samples (need atoms, coordinates, smiles, enriched_description): {lmdb_paths}"
            )
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[TmQMEnrichedLMDBDataset] {len(self.samples)} samples; response=enriched_description")

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
        user_content = f"{INSTRUCTION_STAGE2} {smiles}"
        messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": response}]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(full_text, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_enc = self.tokenizer(prefix_str, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])
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


class Stage2JsonQADataset(Dataset):
    """JSON Q&A (no 3D) for default Stage2."""

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
        prefix_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(full_text, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_enc = self.tokenizer(prefix_str, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])
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
    batch_size = len(id_lists) if id_lists else 0
    max_len = max(len(ids) for ids in id_lists) if id_lists else 0
    if max_len == 0:
        return torch.zeros(batch_size, 0, dtype=dtype), torch.zeros(batch_size, 0, dtype=torch.long)
    padded = torch.full((batch_size, max_len), pad_id, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        length = len(ids)
        padded[i, :length] = torch.tensor(ids, dtype=dtype)
        mask[i, :length] = 1
    return padded, mask


class Stage2MixedCollator:
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


class Stage2Trainer(Trainer):
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
        proj_state = {
            f"single_token_projection.{k}": v.cpu() for k, v in model.single_token_projection_layer.state_dict().items()
        }
        save_file(proj_state, os.path.join(output_dir, SINGLE_TOKEN_PROJECTION_SAFETENSORS))
        torch.save({"model": model.unimol.state_dict()}, os.path.join(output_dir, THREE_D_ENCODER_STATE_PT))
        with open(os.path.join(output_dir, "multimodal_config.json"), "w", encoding="utf-8") as f:
            json.dump({"atom_dim": ATOM_DIM, "single_token_only": True}, f, indent=2)

    def _load_best_model(self):
        ckpt_dir = self.state.best_model_checkpoint
        if ckpt_dir is None:
            return
        model = unwrap_hf_model(self.model)
        model.llm.load_adapter(ckpt_dir, "default")
        st_path = os.path.join(ckpt_dir, SINGLE_TOKEN_PROJECTION_SAFETENSORS)
        if os.path.isfile(st_path):
            raw = load_file(st_path, device="cpu")
            sd = strip_single_token_projection_state_dict(dict(raw))
            if sd:
                model.single_token_projection_layer.load_state_dict(sd, strict=False)
        unimol_pt = os.path.join(ckpt_dir, THREE_D_ENCODER_STATE_PT)
        if os.path.isfile(unimol_pt):
            state = torch.load(unimol_pt, map_location="cpu", weights_only=False)
            if "model" in state:
                state = state["model"]
            model.unimol.load_state_dict(state, strict=False)


class DescriptionMultiTokenTrainer(InferenceOnlyCheckpointMixin, MultimodalMultiTokenTrainer):
    pass


class Description3dOnlyTrainer(InferenceOnlyCheckpointMixin, MultimodalFullTrainer):
    pass


def add_ablation_arguments(parser):
    """Register --ablation and ablation-specific CLI flags on an ArgumentParser."""
    parser.add_argument(
        "--ablation",
        type=str,
        default="stage2",
        choices=ABLATION_CHOICES,
        help="Training mode: stage2 (default), freeze_3d, random_3d, multi_token, 3d_only",
    )
    parser.add_argument(
        "--3D_encoder_ckpt",
        dest="three_d_encoder_ckpt",
        type=str,
        default=PROPERTY_DEFAULTS["3D_encoder_ckpt"],
        help="Pretrained 3D encoder weights (ablation modes except stage2)",
    )
    parser.add_argument(
        "--Stage2_ckpt",
        dest="stage2_ckpt",
        type=str,
        default=PROPERTY_DEFAULTS["Stage2_ckpt"],
        help="Stage2 checkpoint dir; used to resolve 3D encoder if --3D_encoder_ckpt unset",
    )
    parser.add_argument("--save_steps", type=int, default=1000, help="Checkpoint save interval (ablation modes)")
    parser.add_argument("--no_eval", action="store_true", help="Disable validation (freeze_3d / random_3d)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit LMDB samples (multi_token / 3d_only smoke tests)")
    parser.add_argument("--random_3d_seed", type=int, default=DESCRIPTION_DEFAULTS["random_3d_seed"])
    parser.add_argument(
        "--lora_init",
        type=str,
        default="from_scratch",
        choices=["from_scratch", "pretrained"],
        help="LoRA init for multi_token / 3d_only",
    )
    parser.add_argument("--train_3d_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--response_key",
        type=str,
        default=None,
        help="LMDB response field (default: enriched_description for stage2, polished_description for ablations)",
    )


def _training_args_common(args_main, output_dir, *, eval_strategy, eval_steps, ddp_unused, save_kwargs):
    ds_config = os.path.join(_PROJECT_ROOT, "ds_config.json")
    return TrainingArguments(
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
        save_steps=args_main.save_steps,
        save_strategy="steps",
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        report_to="wandb",
        ddp_find_unused_parameters=ddp_unused,
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        save_total_limit=5 if args_main.ablation in ("stage2", "freeze_3d", "random_3d") else 10,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        **save_kwargs,
    )


def _run_stage2_default(args_main, local_rank):
    output_dir = args_main.output_dir or _DEFAULT_OUTPUT["stage2"]
    if local_rank == 0:
        print("[Stage2] Mixed LMDB (enriched_description) + JSON QA; LLM unfrozen (LoRA); 3D=single-token-only")
        print(f"[Stage2] LoRA: r={args_main.lora_r}, alpha={args_main.lora_alpha}, target={args_main.lora_target}")

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = MultimodalModel(
        args_main.model_name,
        args_main.three_d_encoder_dict,
        recipe=RECIPE_STAGE2,
        adapter_path=args_main.stage1_ckpt,
        lora_r=args_main.lora_r,
        lora_alpha=args_main.lora_alpha,
        lora_target=args_main.lora_target,
    )

    lmdb_dataset = Stage2EnrichedLMDBDataset(
        args_main.train_lmdb, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, max_samples=args_main.max_samples
    )
    json_dataset = Stage2JsonQADataset(args_main.json_qa, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH)
    train_dataset = ConcatDataset([lmdb_dataset, json_dataset])
    val_dataset = Stage2EnrichedLMDBDataset(
        [args_main.val_lmdb], tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, max_samples=args_main.max_samples
    )

    if local_rank == 0:
        print(f"Train: LMDB {len(lmdb_dataset)} + JSON {len(json_dataset)} = {len(train_dataset)}")
        print(f"Val: {len(val_dataset)} (LMDB val set)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args_main.epochs,
        deepspeed=os.path.join(_PROJECT_ROOT, "ds_config.json"),
        per_device_train_batch_size=args_main.batch_size,
        per_device_eval_batch_size=args_main.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args_main.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.05,
        logging_steps=10,
        save_steps=1,
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
    trainer = Stage2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=Stage2MixedCollator(tokenizer),
    )
    trainer.train()


def _run_freeze_or_random(args_main, local_rank):
    ablation = args_main.ablation
    require_existing_model_path(args_main.model_name, "--model_name")
    encoder_ckpt = resolve_three_d_encoder_ckpt(args_main.three_d_encoder_ckpt, args_main.stage2_ckpt)
    output_dir = args_main.output_dir or _DEFAULT_OUTPUT[ablation]

    if local_rank == 0:
        tag = "Freeze3D" if ablation == "freeze_3d" else "Random3D"
        print(f"[Description-{tag}] frozen/random structure slot; train LoRA only")
        print(f"[Description-{tag}] 3D encoder weights: {encoder_ckpt}")
        if ablation == "random_3d":
            print(f"[Description-{tag}] random_3d_seed={args_main.random_3d_seed}")

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model_kwargs = dict(
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=encoder_ckpt,
        init_ckpt=None,
        train_3d_encoder=False,
        train_projection=False,
        train_lora=True,
        load_pretrained_projection=False,
        load_pretrained_lora=False,
        lora_r=args_main.lora_r,
        lora_alpha=args_main.lora_alpha,
        lora_target=args_main.lora_target,
    )
    if ablation == "freeze_3d":
        model = MultimodalModelFreeze3D(args_main.model_name, args_main.three_d_encoder_dict, **model_kwargs)
    else:
        model = MultimodalModelRandom3D(
            args_main.model_name,
            args_main.three_d_encoder_dict,
            random_3d_seed=args_main.random_3d_seed,
            **model_kwargs,
        )
    if local_rank == 0:
        model.llm.print_trainable_parameters()

    response_key = args_main.response_key or RESPONSE_KEY
    lmdb_dataset = TmQMEnrichedLMDBDataset(
        args_main.train_lmdb,
        tokenizer=tokenizer,
        response_key=response_key,
        max_samples=args_main.max_samples,
    )
    json_dataset = JsonQADataset(args_main.json_qa, tokenizer=tokenizer)
    train_dataset = ConcatDataset([lmdb_dataset, json_dataset])
    val_dataset = None
    if not args_main.no_eval:
        val_dataset = TmQMEnrichedLMDBDataset(
            [args_main.val_lmdb],
            tokenizer=tokenizer,
            response_key=response_key,
            max_samples=args_main.max_samples,
        )

    if local_rank == 0:
        print(f"Train: LMDB {len(lmdb_dataset)} + JSON {len(json_dataset)} = {len(train_dataset)}")
        print(f"Val: {len(val_dataset) if val_dataset else 'disabled (--no_eval)'}")

    training_args = _training_args_common(
        args_main,
        output_dir,
        eval_strategy="no" if args_main.no_eval else "steps",
        eval_steps=None if args_main.no_eval else 1000,
        ddp_unused=(ablation == "random_3d"),
        save_kwargs=DESCRIPTION_SAVE_KWARGS,
    )
    trainer = DescriptionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MixedCollator(tokenizer),
    )
    trainer.train()


def _run_multi_token(args_main, local_rank):
    require_existing_model_path(args_main.model_name, "--model_name")
    encoder_ckpt = resolve_three_d_encoder_ckpt(args_main.three_d_encoder_ckpt, args_main.stage2_ckpt)
    init_ckpt = args_main.stage2_ckpt if args_main.lora_init == "pretrained" else None
    output_dir = args_main.output_dir or _DEFAULT_OUTPUT["multi_token"]

    if local_rank == 0:
        print(
            f"[Description-multi-token] learnable query tokens; lora_init={args_main.lora_init}; "
            f"prompt=instruction+SMILES; target={args_main.response_key or RESPONSE_KEY}"
        )
        print(f"[Description-multi-token] 3D encoder weights: {encoder_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = MultimodalModelMultiToken(
        args_main.model_name,
        args_main.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=encoder_ckpt,
        init_ckpt=init_ckpt,
        train_3d_encoder=True,
        train_projection=True,
        train_lora=args_main.train_lora,
        lora_r=args_main.lora_r,
        lora_alpha=args_main.lora_alpha,
        lora_target=args_main.lora_target,
        load_pretrained_lora=(args_main.lora_init == "pretrained"),
    )
    if local_rank == 0 and args_main.train_lora:
        model.llm.print_trainable_parameters()

    response_key = args_main.response_key or RESPONSE_KEY
    train_dataset = TmQMEnrichedTokenizedDataset(
        args_main.train_lmdb,
        tokenizer=tokenizer,
        include_smiles=True,
        max_samples=args_main.max_samples,
        response_key=response_key,
    )
    if local_rank == 0:
        print(f"[Description-multi-token] Train samples: {len(train_dataset)}")

    training_args = _training_args_common(
        args_main,
        output_dir,
        eval_strategy="no",
        eval_steps=None,
        ddp_unused=False,
        save_kwargs=DESCRIPTION_SAVE_KWARGS,
    )
    trainer = DescriptionMultiTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()


def _run_3d_only(args_main, local_rank):
    require_existing_model_path(args_main.model_name, "--model_name")
    encoder_ckpt = resolve_three_d_encoder_ckpt(args_main.three_d_encoder_ckpt, args_main.stage2_ckpt)
    init_ckpt = args_main.stage2_ckpt if args_main.lora_init == "pretrained" else None
    output_dir = args_main.output_dir or _DEFAULT_OUTPUT["3d_only"]

    if local_rank == 0:
        print(
            f"[Description-3D-only] prompt=instruction only (NO SMILES); "
            f"target={args_main.response_key or RESPONSE_KEY}"
        )
        print(
            f"[Description-3D-only] trainable: 3D_encoder={args_main.train_3d_encoder}, "
            f"projection={args_main.train_projection}, lora={args_main.train_lora}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = MultimodalModel(
        args_main.model_name,
        args_main.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=encoder_ckpt,
        init_ckpt=init_ckpt,
        train_3d_encoder=args_main.train_3d_encoder,
        train_projection=args_main.train_projection,
        train_lora=args_main.train_lora,
        load_pretrained_projection=False,
        load_pretrained_lora=(args_main.lora_init == "pretrained"),
        lora_r=args_main.lora_r,
        lora_alpha=args_main.lora_alpha,
        lora_target=args_main.lora_target,
    )
    if local_rank == 0 and args_main.train_lora:
        model.llm.print_trainable_parameters()

    response_key = args_main.response_key or RESPONSE_KEY
    train_dataset = TmQMEnrichedTokenizedDataset(
        args_main.train_lmdb,
        tokenizer=tokenizer,
        include_smiles=False,
        max_samples=args_main.max_samples,
        response_key=response_key,
    )
    if local_rank == 0:
        print(f"[Description-3D-only] Train samples: {len(train_dataset)}")

    training_args = _training_args_common(
        args_main,
        output_dir,
        eval_strategy="no",
        eval_steps=None,
        ddp_unused=False,
        save_kwargs=DESCRIPTION_SAVE_KWARGS,
    )
    trainer = Description3dOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()


def run_training(args_main):
    """Dispatch training by --ablation."""
    try:
        import multiprocessing

        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.init(project=_WANDB_PROJECT[args_main.ablation])

    if args_main.ablation == "stage2":
        _run_stage2_default(args_main, local_rank)
    elif args_main.ablation in ("freeze_3d", "random_3d"):
        _run_freeze_or_random(args_main, local_rank)
    elif args_main.ablation == "multi_token":
        _run_multi_token(args_main, local_rank)
    elif args_main.ablation == "3d_only":
        _run_3d_only(args_main, local_rank)
    else:
        raise ValueError(f"Unknown ablation={args_main.ablation!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage2: mixed LMDB + JSON QA, or description ablation (--ablation)"
    )
    parser.add_argument("--model_name", type=str, default=STAGE2_DEFAULTS["model_name"])
    parser.add_argument(
        "--Stage1_ckpt",
        dest="stage1_ckpt",
        type=str,
        default=STAGE2_DEFAULTS["Stage1_ckpt"],
        help="Stage1 single-token-only checkpoint (default stage2 mode only)",
    )
    parser.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=STAGE2_DEFAULTS["3D_encoder_dict"],
    )
    parser.add_argument("--train_lmdb", nargs="+", default=STAGE2_DEFAULTS["train_lmdb"])
    parser.add_argument("--val_lmdb", type=str, default=STAGE2_DEFAULTS["val_lmdb"])
    parser.add_argument("--json_qa", nargs="+", default=STAGE2_DEFAULTS["json_qa"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=STAGE2_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=STAGE2_DEFAULTS["lora_alpha"])
    parser.add_argument(
        "--lora_target",
        type=str,
        default=STAGE2_DEFAULTS["lora_target"],
        choices=["qv", "qkv", "all"],
    )
    parser.add_argument("--epochs", type=int, default=STAGE2_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=STAGE2_DEFAULTS["lr"])
    parser.add_argument("--batch_size", type=int, default=STAGE2_DEFAULTS["batch_size"])
    parser.add_argument("--local_rank", type=int, default=-1)

    add_ablation_arguments(parser)
    args_main = parser.parse_args()

    if args_main.ablation != "stage2":
        if args_main.model_name == STAGE2_DEFAULTS["model_name"]:
            args_main.model_name = DESCRIPTION_DEFAULTS["model_name"]
        if args_main.three_d_encoder_dict == STAGE2_DEFAULTS["3D_encoder_dict"]:
            args_main.three_d_encoder_dict = DESCRIPTION_DEFAULTS["3D_encoder_dict"]
        if args_main.epochs == STAGE2_DEFAULTS["epochs"] and DESCRIPTION_DEFAULTS["epochs"] != STAGE2_DEFAULTS["epochs"]:
            args_main.epochs = DESCRIPTION_DEFAULTS["epochs"]
        if args_main.batch_size == STAGE2_DEFAULTS["batch_size"] and DESCRIPTION_DEFAULTS["batch_size"] != STAGE2_DEFAULTS["batch_size"]:
            args_main.batch_size = DESCRIPTION_DEFAULTS["batch_size"]

    run_training(args_main)
