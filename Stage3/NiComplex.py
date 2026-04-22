"""
SFT training for ΔΔG prediction on a Ni(III) intermediate dataset (`/data/jingyuan_data/NiComplex`),
using the same 3D encoder + single-token + Qwen3-4B (4bit + LoRA) architecture and training setup
as `SFT_ligandscreen_enantio_unimol_full.py`.

Differences vs LigandScreen script:
- Data: merge NiComplex LMDBs (default path if --lmdb omitted; see --help),
  shuffle with --split_seed, split 80/10/10 train/val/test.
  Checkpoints: --output_dir/seed_<split_seed>/.
- Keys per record: "atoms", "coordinates", "ddG", optional "smiles", "temp" (K). Target: ΔΔG only.
"""

import os
import sys
import json
import pickle
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrainingArguments
import wandb

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from multimodal_LLM import RECIPE_STAGE3, MultimodalModel
from utils import (
    MultimodalCollator,
    MultimodalFullTrainer,
    MAX_SEQ_LENGTH,
    _atoms_coords_remove_h_center,
    format_instruction_field,
    tokenize_generation_sample_object_ref,
)
from train_defaults import NICOMPLEX_DEFAULTS

NI_INSTRUCTION = "The key Ni(III) intermediate complex, bearing chiral ligands and carbon groups, determines the enantioselectivity (\u2206\u2206G) of nickel-catalyzed cross-coupling reactions. What is the \u2206\u2206G (in kcal/mol) for this reaction? Given the Ni(III) intermediate {smiles} and {temp} K, respond with the numerical value only:"  # backward-compatible alias


def _lmdb_env_kwargs():
    return dict(
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=256,
    )


def load_merged_valid_nicomplex_records(lmdb_paths, local_rank=0):
    """Merge LMDBs and return records with valid ddG (same filtering as the dataset)."""
    all_raw = []
    for p in lmdb_paths:
        if not os.path.exists(p):
            if local_rank == 0:
                print(f"[Stage3_NiComplex] skip missing LMDB: {p}")
            continue
        all_raw.extend(read_nicomplex_lmdb(p, max_samples=None))
    merged_valid = []
    nan_skipped = 0
    for d in all_raw:
        if "atoms" not in d or "coordinates" not in d or "ddG" not in d:
            continue
        try:
            val = float(d["ddG"])
            if np.isnan(val):
                nan_skipped += 1
                continue
        except (ValueError, TypeError):
            nan_skipped += 1
            continue
        merged_valid.append(d)
    return merged_valid, nan_skipped


def read_nicomplex_lmdb(lmdb_path, max_samples=None, show_progress=True):
    """LMDB reader for NiComplex dataset (pickle-serialized dicts)."""
    import tqdm as _tqdm

    env = lmdb.open(lmdb_path, **_lmdb_env_kwargs())
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
        print(f"[Stage3_NiComplex-LMDB] skipped {bad} bad entries")
    return data_list


class NiComplexDDGDataset(Dataset):
    """
    Dataset for ΔΔG prediction using 3D structures only.

    Each sample:
        - Input: instruction (no SMILES) + 3D block (atoms, coordinates)
        - Output (generation mode): numeric ΔΔG value as string.
        - Output (regression mode): numeric ΔΔG value (optionally normalized).
    """

    def __init__(
        self,
        lmdb_paths=None,
        tokenizer=None,
        max_length=512,
        max_samples=None,
        prop_mean=None,
        prop_std=None,
        instruction=None,
        samples=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prop_mean = prop_mean
        self.prop_std = prop_std
        self.property_key = "ddG"
        self.instruction = instruction or NI_INSTRUCTION
        # If instruction contains "{smiles}" / "{temp}", __getitem__ fills from d["smiles"], d["temp"].

        self.samples = []
        nan_skipped = 0
        if samples is not None:
            data_iter = samples
        else:
            data_iter = []
            for p in (lmdb_paths or []):
                if not os.path.exists(p):
                    continue
                raw = read_nicomplex_lmdb(p, max_samples=max_samples)
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
            self.samples.append(d)

        if not self.samples and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            src = lmdb_paths if samples is None else "provided samples"
            raise RuntimeError(f"No valid samples: {src}")
        if nan_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Stage3_NiComplexDataset] Skipped {nan_skipped} samples with NaN {self.property_key}")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"[Stage3_NiComplexDataset] property={self.property_key}, {len(self.samples)} samples; "
                f"mode=GENERATION"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = self.samples[idx]
        atoms = d["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(d["coordinates"], dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
        value = float(d[self.property_key])

        target_text = f"{value:.6f}"

        # Chat prompt: instruction may include SMILES and temp (K) from LMDB.
        smiles = format_instruction_field(d.get("smiles", ""))
        temp = format_instruction_field(d.get("temp", ""))
        if "{smiles}" in self.instruction or "{temp}" in self.instruction:
            user_content = self.instruction.format(smiles=smiles, temp=temp)
        else:
            user_content = self.instruction
        ids = tokenize_generation_sample_object_ref(self.tokenizer, user_content, target_text)
        return {**ids, "atoms": atoms, "coordinates": coords}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage3_NiComplex: Ni(III) ΔΔG from 3D structures only"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=NICOMPLEX_DEFAULTS["model_name"],
    )
    parser.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=NICOMPLEX_DEFAULTS["3D_encoder_dict"],
    )
    parser.add_argument(
        "--Stage2_ckpt",
        dest="stage2_ckpt",
        type=str,
        default=NICOMPLEX_DEFAULTS["Stage2_ckpt"],
    )
    parser.add_argument(
        "--3D_encoder_ckpt",
        dest="three_d_encoder_ckpt",
        type=str,
        default=NICOMPLEX_DEFAULTS["3D_encoder_ckpt"],
        help="Optional override for 3D encoder ckpt. If set, it takes priority over Stage2_ckpt/3D_encoder.pt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=NICOMPLEX_DEFAULTS["output_dir"],
        help="Base directory; checkpoints go to <output_dir>/seed_<split_seed>/.",
    )
    parser.add_argument(
        "--lmdb",
        action="append",
        default=None,
        dest="lmdb_paths",
        metavar="PATH",
        help=(
            "NiComplex LMDB path; pass multiple times for multiple files. "
            "If omitted, uses defaults from train_defaults.py."
        ),
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=NICOMPLEX_DEFAULTS["split_seed"],
        help=(
            "Random seed for 80/10/10 train/val/test shuffle and TrainingArguments.seed. "
            "Checkpoint directory: --output_dir/seed_<split_seed>/."
        ),
    )
    parser.add_argument("--lora_r", type=int, default=NICOMPLEX_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=NICOMPLEX_DEFAULTS["lora_alpha"])
    parser.add_argument("--lora_target", type=str, default=NICOMPLEX_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"])
    parser.add_argument(
        "--projection_init",
        type=str,
        default=NICOMPLEX_DEFAULTS["projection_init"],
        choices=["pretrained", "from_scratch"],
        help="Initialize projection from Stage2 ckpt or from scratch.",
    )
    parser.add_argument("--epochs", type=int, default=NICOMPLEX_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=NICOMPLEX_DEFAULTS["lr"])
    parser.add_argument("--batch_size", type=int, default=NICOMPLEX_DEFAULTS["batch_size"])
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    # deepspeed / torchrun will pass --local_rank; accept it to avoid argparse error
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    try:
        import multiprocessing

        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    # fixed design: generation-only + single-token-only projection

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    split_seed = args.split_seed
    lmdb_paths = args.lmdb_paths or NICOMPLEX_DEFAULTS["lmdb"]

    merged_valid, nan_skipped = load_merged_valid_nicomplex_records(lmdb_paths, local_rank=local_rank)

    n_total = len(merged_valid)
    if n_total == 0:
        raise RuntimeError(f"[Stage3_NiComplex] No valid ddG samples after merging LMDBs: {lmdb_paths}")
    if nan_skipped > 0 and local_rank == 0:
        print(f"[Stage3_NiComplex] Skipped {nan_skipped} invalid ddG entries while merging")

    indices = np.arange(n_total)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(indices)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    train_samples = [merged_valid[i] for i in train_idx]
    val_samples = [merged_valid[i] for i in val_idx]
    _test_samples = [merged_valid[i] for i in test_idx]

    if args.max_train_samples is not None:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_eval_samples is not None:
        val_samples = val_samples[: args.max_eval_samples]

    run_output_dir = os.path.join(args.output_dir, f"seed_{split_seed}")
    if local_rank == 0:
        os.makedirs(run_output_dir, exist_ok=True)
        print(
            f"[Stage3_NiComplex] split_seed={split_seed} | total={n_total}, "
            f"train={len(train_samples)}, val={len(val_samples)}, test={len(_test_samples)} | "
            f"output_dir={run_output_dir}"
        )

    if local_rank == 0:
        wandb.init(
            project="Stage3_NiComplex",
            name=f"seed_{split_seed}",
            reinit=True,
            config=vars(args),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalModel(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=args.three_d_encoder_ckpt,
        init_ckpt=args.stage2_ckpt,
        load_pretrained_projection=(args.projection_init == "pretrained"),
        train_3d_encoder=True,
        train_projection=True,
        train_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
    )

    train_dataset = NiComplexDDGDataset(
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
        samples=train_samples,
        instruction=NI_INSTRUCTION,
    )
    eval_dataset = NiComplexDDGDataset(
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
        samples=val_samples,
        instruction=NI_INSTRUCTION,
    )

    label_names = ["labels"]
    data_collator = MultimodalCollator(tokenizer)
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.05,
        logging_steps=100,
        save_steps=100,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=2000,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        label_names=label_names,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        save_total_limit=10,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=split_seed,
    )

    trainer = MultimodalFullTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(run_output_dir)

    if local_rank == 0:
        with open(os.path.join(run_output_dir, "split_seed.txt"), "w", encoding="utf-8") as f:
            f.write(f"{split_seed}\n")
        wandb.finish()

