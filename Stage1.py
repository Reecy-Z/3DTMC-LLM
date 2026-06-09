"""
Stage1 SFT: frozen LLM + trainable 3D encoder (+ projection).

Default: instruction + SMILES -> description (single-token projection).

Ablation modes (--ablation):
  3d_only      3D structure slot only (no SMILES in prompt)
  multi_token  Learnable multi-token query projection instead of single-token projection

Usage:
  deepspeed --num_gpus=2 Stage1.py
  deepspeed --num_gpus=2 Stage1.py --ablation 3d_only --3D_encoder_ckpt /path/to/encoder
  deepspeed --num_gpus=2 Stage1.py --ablation multi_token --3D_encoder_ckpt /path/to/encoder
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import wandb
from safetensors.torch import save_file
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer

import utils  # noqa: F401
from utils import (
    ATOM_DIM,
    MAX_SEQ_LENGTH,
    SINGLE_TOKEN_PROJECTION_SAFETENSORS,
    THREE_D_ENCODER_STATE_PT,
    MultimodalCollator,
    _atoms_coords_remove_h_center,
    read_lmdb,
    tokenize_generation_sample_object_ref,
    unwrap_hf_model,
)
from multimodal_LLM import (
    MULTI_TOKEN_PROJECTION_PT,
    NUM_STRUCTURE_QUERIES,
    RECIPE_STAGE1,
    MultimodalModel,
    MultimodalModelMultiToken,
    MultimodalMultiTokenTrainer,
)
from train_defaults import STAGE1_DEFAULTS

ABLATION_CHOICES = ("stage1", "3d_only", "multi_token")

INSTRUCTION_STAGE1 = "Give me a description of this transition metal complex:"
INSTRUCTION_3D_ONLY = "Give a description of this transition metal complex:"
RESPONSE_KEY_DEFAULT = "description"

_WANDB_PROJECT = {
    "stage1": "Stage1",
    "3d_only": "Stage1_ablation_3d_only",
    "multi_token": "Stage1_ablation_multi_token",
}

_DEFAULT_OUTPUT = {
    "stage1": STAGE1_DEFAULTS["output_dir"],
    "3d_only": "Stage1_3d_only_ckpt",
    "multi_token": "Stage1_multi_token_ckpt",
}


class TmQMDescriptionDataset(Dataset):
    """Stage1 LMDB: instruction (+ optional SMILES) + 3D -> description."""

    def __init__(
        self,
        lmdb_paths,
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
        max_samples=None,
        *,
        include_smiles=True,
        instruction=None,
        response_key=RESPONSE_KEY_DEFAULT,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_smiles = include_smiles
        self.instruction = instruction or INSTRUCTION_STAGE1
        self.response_key = response_key
        self.samples = []
        for path in lmdb_paths:
            if not os.path.exists(path):
                continue
            raw = read_lmdb(path, max_samples=max_samples)
            for data in raw:
                if "atoms" not in data or "coordinates" not in data or "smiles" not in data:
                    continue
                response = data.get(response_key)
                if not response or (isinstance(response, str) and not response.strip()):
                    continue
                self.samples.append(data)
        if not self.samples and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            raise RuntimeError(
                f"No valid samples (need atoms, coordinates, smiles, {response_key}): {lmdb_paths}"
            )
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            mode = "instruction+SMILES" if include_smiles else "3D-only (no SMILES)"
            print(f"[Stage1-LMDB] {len(self.samples)} samples; response={response_key}; mode={mode}")

    def __len__(self):
        return len(self.samples)

    def _user_content(self, smiles: str) -> str:
        if self.include_smiles:
            return f"{self.instruction} {smiles}"
        return self.instruction.strip()

    def __getitem__(self, idx):
        data = self.samples[idx]
        response = data[self.response_key]
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
        user_content = self._user_content(data["smiles"])
        ids = tokenize_generation_sample_object_ref(
            self.tokenizer,
            user_content,
            response,
            prefix_len_mode="prefix",
            tokenizer_full=dict(truncation=True, max_length=self.max_length),
            tokenizer_split_parts=dict(truncation=True, max_length=self.max_length),
        )
        return {**ids, "atoms": atoms, "coordinates": coords}


class Stage1Trainer(Trainer):
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
        if getattr(model, "single_token_projection_layer", None) is not None:
            proj_state = {
                f"single_token_projection.{k}": v.cpu()
                for k, v in model.single_token_projection_layer.state_dict().items()
            }
            save_file(proj_state, os.path.join(output_dir, SINGLE_TOKEN_PROJECTION_SAFETENSORS))
        torch.save({"model": model.unimol.state_dict()}, os.path.join(output_dir, THREE_D_ENCODER_STATE_PT))
        with open(os.path.join(output_dir, "multimodal_config.json"), "w", encoding="utf-8") as f:
            json.dump({"atom_dim": ATOM_DIM, "single_token_only": True}, f, indent=2)


class Stage1MultiTokenTrainer(Stage1Trainer):
    """Stage1 checkpoint layout with multi_token_projection.pt (no LoRA weights)."""

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)
        model = unwrap_hf_model(self.model)
        model.tokenizer.save_pretrained(output_dir)
        if getattr(model, "multi_token_projection_layer", None) is not None:
            state = {k: v.cpu() for k, v in model.multi_token_projection_layer.state_dict().items()}
            torch.save(state, os.path.join(output_dir, MULTI_TOKEN_PROJECTION_PT))
        torch.save({"model": model.unimol.state_dict()}, os.path.join(output_dir, THREE_D_ENCODER_STATE_PT))
        with open(os.path.join(output_dir, "multimodal_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "atom_dim": ATOM_DIM,
                    "structure_tokens": "multi_token_query",
                    "num_structure_queries": NUM_STRUCTURE_QUERIES,
                    "include_single_token": False,
                    "single_token_only": False,
                },
                f,
                indent=2,
            )


def add_ablation_arguments(parser):
    parser.add_argument(
        "--ablation",
        type=str,
        default="stage1",
        choices=ABLATION_CHOICES,
        help="Training mode: stage1 (default), 3d_only, multi_token",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Limit LMDB samples (smoke tests)")
    parser.add_argument(
        "--response_key",
        type=str,
        default=RESPONSE_KEY_DEFAULT,
        help="LMDB assistant target field (default: description)",
    )


def _build_datasets(args_main, tokenizer):
    include_smiles = args_main.ablation != "3d_only"
    instruction = INSTRUCTION_STAGE1 if include_smiles else INSTRUCTION_3D_ONLY
    kwargs = dict(
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
        max_samples=args_main.max_samples,
        include_smiles=include_smiles,
        instruction=instruction,
        response_key=args_main.response_key,
    )
    train_dataset = TmQMDescriptionDataset(args_main.train_lmdb, **kwargs)
    val_dataset = TmQMDescriptionDataset([args_main.val_lmdb], **kwargs)
    return train_dataset, val_dataset


def _training_args(args_main, output_dir):
    return TrainingArguments(
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
        save_steps=args_main.save_steps,
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


def _run_stage1_default(args_main, local_rank):
    output_dir = args_main.output_dir or _DEFAULT_OUTPUT["stage1"]
    if local_rank == 0:
        print("[Stage1] description + 3D encoder (single-token projection); LLM frozen; prompt=instruction+SMILES")

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = MultimodalModel(
        args_main.model_name,
        args_main.three_d_encoder_dict,
        recipe=RECIPE_STAGE1,
        three_d_encoder_ckpt=args_main.three_d_encoder_ckpt,
    )
    train_dataset, val_dataset = _build_datasets(args_main, tokenizer)
    if local_rank == 0:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    trainer = Stage1Trainer(
        model=model,
        args=_training_args(args_main, output_dir),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()


def _run_3d_only(args_main, local_rank):
    output_dir = args_main.output_dir or _DEFAULT_OUTPUT["3d_only"]
    if local_rank == 0:
        print("[Stage1-3D-only] frozen LLM; train 3D encoder + single-token projection; prompt=instruction only (NO SMILES)")

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = MultimodalModel(
        args_main.model_name,
        args_main.three_d_encoder_dict,
        recipe=RECIPE_STAGE1,
        three_d_encoder_ckpt=args_main.three_d_encoder_ckpt,
    )
    train_dataset, val_dataset = _build_datasets(args_main, tokenizer)
    if local_rank == 0:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    trainer = Stage1Trainer(
        model=model,
        args=_training_args(args_main, output_dir),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()


def _run_multi_token(args_main, local_rank):
    output_dir = args_main.output_dir or _DEFAULT_OUTPUT["multi_token"]
    if local_rank == 0:
        print(
            "[Stage1-multi_token] frozen LLM; train 3D encoder + multi-token query projection; "
            "prompt=instruction+SMILES"
        )

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = MultimodalModelMultiToken(
        args_main.model_name,
        args_main.three_d_encoder_dict,
        recipe=RECIPE_STAGE1,
        three_d_encoder_ckpt=args_main.three_d_encoder_ckpt,
        train_3d_encoder=True,
        train_projection=True,
        train_lora=False,
        load_pretrained_lora=False,
    )
    train_dataset, val_dataset = _build_datasets(args_main, tokenizer)
    if local_rank == 0:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    trainer = Stage1MultiTokenTrainer(
        model=model,
        args=_training_args(args_main, output_dir),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()


def run_training(args_main):
    try:
        import multiprocessing

        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.init(project=_WANDB_PROJECT[args_main.ablation])

    if args_main.ablation == "stage1":
        _run_stage1_default(args_main, local_rank)
    elif args_main.ablation == "3d_only":
        _run_3d_only(args_main, local_rank)
    elif args_main.ablation == "multi_token":
        _run_multi_token(args_main, local_rank)
    else:
        raise ValueError(f"Unknown ablation={args_main.ablation!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage1: frozen LLM + 3D encoder projection, or ablation (--ablation)"
    )
    parser.add_argument("--model_name", type=str, default=STAGE1_DEFAULTS["model_name"])
    parser.add_argument(
        "--3D_encoder_ckpt",
        dest="three_d_encoder_ckpt",
        type=str,
        default=STAGE1_DEFAULTS["3D_encoder_ckpt"],
        help="Pretrained 3D encoder weights (.pt or directory with model.safetensors)",
    )
    parser.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=STAGE1_DEFAULTS["3D_encoder_dict"],
    )
    parser.add_argument("--train_lmdb", nargs="+", default=STAGE1_DEFAULTS["train_lmdb"])
    parser.add_argument("--val_lmdb", type=str, default=STAGE1_DEFAULTS["val_lmdb"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=STAGE1_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=STAGE1_DEFAULTS["lr"])
    parser.add_argument("--batch_size", type=int, default=STAGE1_DEFAULTS["batch_size"])
    parser.add_argument("--save_steps", type=int, default=STAGE1_DEFAULTS["save_steps"])
    parser.add_argument("--local_rank", type=int, default=-1)

    add_ablation_arguments(parser)
    args_main = parser.parse_args()
    run_training(args_main)
