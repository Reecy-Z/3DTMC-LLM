"""
Fine-tune Uni-Mol + Qwen3-4B on Vaska's complexes for dihydrogen activation
energy barrier, using 3D structures (atoms + coordinates) and SMILES in the prompt.

Data source:
    /data/jingyuan_data/vaskas-space/data.lmdb (default; override with --lmdb)

LMDB entries are expected to contain:
    - "atoms": list/array of atomic symbols
    - "coordinates": (N, 3) float array
    - "barrier": scalar energy barrier (kcal/mol)
    - "smiles": non-empty string (records without SMILES are skipped)
"""

import os
import json
import pickle
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrainingArguments
import wandb

from bos_unimol_qwen_model import BosUnimolQwenModel, RECIPE_STAGE3
from utils import (
    BridgeUnimolCollator,
    BridgeUnimolFullTrainer,
    _atoms_coords_remove_h_center,
    format_instruction_field,
    tokenize_generation_sample_object_ref,
)
from train_defaults import VASKA_DEFAULTS

VASKA_INSTRUCTION = (
    "What is the dihydrogen activation energy barrier (in kcal/mol) of this Vaska's complex? "
    "Given the SMILES {smiles} and the 3D structure, respond with the numerical value only:"
)


def _vaska_lmdb_env_kwargs():
    return dict(
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=256,
    )


def read_vaska_lmdb(lmdb_path, max_samples=None, show_progress=True):
    """LMDB reader for Vaska dataset with robust pickle handling."""
    import tqdm as _tqdm

    env = lmdb.open(lmdb_path, **_vaska_lmdb_env_kwargs())
    txn = env.begin()
    cursor = txn.cursor()
    data_list = []
    stat = env.stat()
    total = min(max_samples, stat["entries"]) if max_samples else stat["entries"]
    desc = f"LMDB {os.path.basename(lmdb_path)}"
    rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0
    it = (
        _tqdm.tqdm(cursor, total=total, desc=desc, unit="samples", mininterval=1.0)
        if (show_progress and rank0)
        else cursor
    )
    bad = 0
    for count, (_, value) in enumerate(it):
        try:
            obj = pickle.loads(value)
        except Exception:
            bad += 1
            continue
        data_list.append(obj)
        if max_samples and (count + 1) >= max_samples:
            break
    env.close()
    if bad > 0 and rank0:
        print(f"[Vaska-LMDB] Skipped {bad} entries that failed to unpickle")
    return data_list


class VaskaBarrierDataset(Dataset):
    """Vaska barrier: 3D block + SMILES in the chat prompt (instruction uses ``{smiles}``)."""

    def __init__(
        self,
        lmdb_paths=None,
        tokenizer=None,
        max_samples=None,
        instruction=None,
        samples=None,
    ):
        self.tokenizer = tokenizer
        self.instruction = instruction or VASKA_INSTRUCTION
        self.property_key = "barrier"

        self.samples = []
        nan_skipped = 0
        smiles_skipped = 0
        # data source: either provided samples list or LMDB paths
        data_iter = None
        if samples is not None:
            data_iter = samples
        else:
            data_iter = []
            for p in (lmdb_paths or []):
                if not os.path.exists(p):
                    continue
                raw = read_vaska_lmdb(p, max_samples=max_samples)
                data_iter.extend(raw)

        for d in data_iter:
            if "atoms" not in d or "coordinates" not in d or self.property_key not in d:
                continue
            try:
                val = float(d[self.property_key])
                if np.isnan(val):
                    nan_skipped += 1
                    continue
            except (ValueError, TypeError):
                nan_skipped += 1
                continue
            sm = format_instruction_field(d.get("smiles"))
            if not sm:
                smiles_skipped += 1
                continue
            self.samples.append(d)

        if not self.samples and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            src = lmdb_paths if samples is None else "provided samples"
            raise RuntimeError(f"No valid samples: {src}")
        if nan_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[VaskaDataset] Skipped {nan_skipped} samples with NaN {self.property_key}")
        if smiles_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[VaskaDataset] Skipped {smiles_skipped} samples (missing SMILES)")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[VaskaDataset] property={self.property_key}, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        barrier = float(data[self.property_key])
        response = str(barrier)

        atoms = data["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(data["coordinates"], dtype=np.float32)
        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)

        smiles = format_instruction_field(data.get("smiles"))
        if "{smiles}" in self.instruction:
            user_content = self.instruction.format(smiles=smiles)
        else:
            user_content = f"{self.instruction} {smiles}".strip()

        ids = tokenize_generation_sample_object_ref(self.tokenizer, user_content, response)
        return {**ids, "atoms": atoms, "coordinates": coords}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SFT_vaska_barrier_unimol_full: Vaska barrier from 3D + SMILES in prompt"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=VASKA_DEFAULTS["model_name"],
    )
    parser.add_argument(
        "--unimol_ckpt",
        type=str,
        default=VASKA_DEFAULTS["unimol_ckpt"],
        help="Uni-Mol checkpoint path (used when init_ckpt is None).",
    )
    parser.add_argument(
        "--unimol_dict",
        type=str,
        default=VASKA_DEFAULTS["unimol_dict"],
    )
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=VASKA_DEFAULTS["init_ckpt"],
        help="Optional checkpoint containing unimol + projection + bridge + LoRA.",
    )
    parser.add_argument(
        "--lmdb",
        type=str,
        default=VASKA_DEFAULTS["lmdb"],
        help="Vaska barrier LMDB path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=VASKA_DEFAULTS["output_dir"],
        help="Base directory; checkpoints go to <output_dir>/seed_<split_seed>/.",
    )
    # BOS-only projection only
    # LoRA hyper-parameters
    parser.add_argument(
        "--lora_r",
        type=int,
        default=VASKA_DEFAULTS["lora_r"],
        help="LoRA rank (default 8).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=VASKA_DEFAULTS["lora_alpha"],
        help="LoRA alpha (default 32).",
    )
    parser.add_argument(
        "--lora_target",
        type=str,
        default=VASKA_DEFAULTS["lora_target"],
        choices=["qv", "qkv", "all"],
        help="LoRA target modules.",
    )
    # training hyper-parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=VASKA_DEFAULTS["epochs"],
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=VASKA_DEFAULTS["lr"],
        help="Learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=VASKA_DEFAULTS["batch_size"],
        help="Per-device train batch size.",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=VASKA_DEFAULTS["split_seed"],
        help=(
            "Random seed for 20/40/40 train/val/test shuffle and TrainingArguments.seed. "
            "Checkpoint directory: --output_dir/seed_<split_seed>/."
        ),
    )

    args_main = parser.parse_args()

    try:
        import multiprocessing

        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    split_seed = args_main.split_seed

    if local_rank == 0:
        print(f"[SFT-Vaska] property=barrier (kcal/mol), MODE: GENERATION, BOS_ONLY (scheme C)")
        print(
            f"[SFT-Vaska] split_seed={split_seed} | checkpoint dir: "
            f"{os.path.join(args_main.output_dir, f'seed_{split_seed}')}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)

    # Load LMDB once; 20/40/40 train/val/test split uses split_seed for shuffle
    all_raw = read_vaska_lmdb(args_main.lmdb, max_samples=None)
    n_total = len(all_raw)
    if n_total == 0:
        raise RuntimeError(f"[SFT-Vaska] LMDB {args_main.lmdb} has no samples")

    data_collator = BridgeUnimolCollator(tokenizer, regression_mode=False)
    ds_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config.json")
    label_names = ["labels"]

    if local_rank == 0:
        print(f"\n[SFT-Vaska] ========== split_seed={split_seed} ==========")

    indices = np.arange(n_total)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(indices)
    n_train = int(0.2 * n_total)
    n_val = int(0.4 * n_total)
    n_test = n_total - n_train - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    train_samples = [all_raw[i] for i in train_idx]
    val_samples = [all_raw[i] for i in val_idx]
    _test_samples = [all_raw[i] for i in test_idx]

    run_output_dir = os.path.join(args_main.output_dir, f"seed_{split_seed}")
    if local_rank == 0:
        os.makedirs(run_output_dir, exist_ok=True)

    model = BosUnimolQwenModel(
        args_main.model_name,
        args_main.unimol_dict,
        recipe=RECIPE_STAGE3,
        unimol_ckpt=args_main.unimol_ckpt,
        init_ckpt=args_main.init_ckpt,
        train_unimol=True,
        train_projection=True,
        train_lora=True,
        lora_r=args_main.lora_r,
        lora_alpha=args_main.lora_alpha,
        lora_target=args_main.lora_target,
    )

    train_dataset = VaskaBarrierDataset(
        tokenizer=tokenizer,
        instruction=None,
        samples=train_samples,
    )
    val_dataset = VaskaBarrierDataset(
        tokenizer=tokenizer,
        instruction=None,
        samples=val_samples,
    )

    if local_rank == 0:
        print(
            f"[SFT-Vaska] seed={split_seed} | total={n_total}, "
            f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(_test_samples)} | "
            f"output_dir={run_output_dir}"
        )

    if local_rank == 0:
        wandb.init(
            project="sft_vaska_barrier_unimol_full",
            name=f"seed_{split_seed}",
            reinit=True,
            config=vars(args_main),
        )

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=args_main.epochs,
        deepspeed=ds_config,
        per_device_train_batch_size=args_main.batch_size,
        per_device_eval_batch_size=args_main.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args_main.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=1000,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        label_names=label_names,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        save_total_limit=20,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=split_seed,
    )

    trainer = BridgeUnimolFullTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(run_output_dir)

    if local_rank == 0:
        with open(os.path.join(run_output_dir, "split_seed.txt"), "w", encoding="utf-8") as f:
            f.write(f"{split_seed}\n")
        wandb.finish()

