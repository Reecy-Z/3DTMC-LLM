#!/usr/bin/env python3
"""
Uni-Mol pretraining with HuggingFace Transformers Trainer and DeepSpeed.
"""

from __future__ import annotations

import os
import sys
import argparse
import pickle
import random
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from scipy.spatial import distance_matrix
from tqdm import tqdm

from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import wandb

# Project paths on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UNIMOL_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PROJECT_ROOT = os.path.dirname(_UNIMOL_ROOT)
for p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Uni-Core")]:
    if p not in sys.path:
        sys.path.insert(0, p)

if _UNIMOL_ROOT not in sys.path:
    sys.path.insert(0, _UNIMOL_ROOT)

# Pretraining loss normalization (aligned with OMol25_MC_all)
DIST_MEAN = 6.590849597363901
DIST_STD = 2.794560980830973

# Atomic number to element symbol (RDKit periodic table)
from rdkit.Chem import GetPeriodicTable
_PERIODIC_TABLE = GetPeriodicTable()


def atoms_to_symbols(atoms: list) -> list[str]:
    """Convert atom entries to element symbols (integers or strings)."""
    result = []
    for a in atoms:
        if isinstance(a, (int, np.integer)):
            result.append(_PERIODIC_TABLE.GetElementSymbol(int(a)))
        else:
            s = str(a).strip()
            if s.isdigit():
                result.append(_PERIODIC_TABLE.GetElementSymbol(int(s)))
            else:
                result.append(s)
    return result


def remove_hydrogen(atoms: list[str], coordinates: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Strip hydrogen atoms."""
    atoms_arr = np.array(atoms)
    coords = np.asarray(coordinates, dtype=np.float32)
    mask_hydrogen = atoms_arr != "H"
    atoms_no_h = atoms_arr[mask_hydrogen].tolist()
    coordinates_no_h = coords[mask_hydrogen]
    return atoms_no_h, coordinates_no_h


def center_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Center coordinates at the origin (subtract centroid)."""
    coords = np.asarray(coordinates, dtype=np.float32)
    return coords - coords.mean(axis=0)


def preprocess_molecule(
    atoms: list[str],
    coordinates: np.ndarray,
    remove_h: bool = True,
    center: bool = True,
    max_atoms: int = 256,
) -> tuple[list[str], np.ndarray]:
    """Preprocess: optional dehydrogenation, centering, and max-atom cap."""
    atoms_processed = atoms
    coordinates_processed = coordinates.copy()

    # 1. Remove hydrogens
    if remove_h:
        atoms_processed, coordinates_processed = remove_hydrogen(atoms_processed, coordinates_processed)

    # 2. Center
    if center and len(coordinates_processed) > 0:
        coordinates_processed = center_coordinates(coordinates_processed)

    # 3. Drop if too many atoms
    if max_atoms is not None and len(atoms_processed) > max_atoms:
        return None, None

    return atoms_processed, coordinates_processed


class LMDBPretrainDataset(Dataset):
    """LMDB dataset for Uni-Mol pretraining."""

    def __init__(
        self,
        path: str,
        remove_hydrogen: bool = True,
        center_coords: bool = True,
        max_atoms: int = 256,
    ):
        self.path = os.path.abspath(path)
        self.remove_hydrogen = remove_hydrogen
        self.center_coords = center_coords
        self.max_atoms = max_atoms

        # Collect LMDB paths
        self._lmdb_paths: list[str] = []
        if os.path.isfile(self.path):
            self._lmdb_paths.append(self.path)
        elif os.path.isdir(self.path):
            for f in sorted(os.listdir(self.path)):
                if f.endswith(".lmdb") or (f.startswith("data") and f.endswith(".lmdb")):
                    self._lmdb_paths.append(os.path.join(self.path, f))
        else:
            raise FileNotFoundError(f"LMDB path does not exist: {self.path}")
        if not self._lmdb_paths:
            raise FileNotFoundError(f"No .lmdb files found under {self.path}")

        # Pre-index (file_idx, key) for all entries
        self._keys: list[tuple[int, bytes]] = []
        for fi, lmdb_path in enumerate(self._lmdb_paths):
            env = lmdb.open(
                lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
            with env.begin() as txn:
                keys = list(txn.cursor().iternext(values=False))
            env.close()
            self._keys.extend((fi, k) for k in keys)

        # Lazy-open envs per worker to avoid multiprocessing issues
        self._envs: list[lmdb.Environment | None] = [None] * len(self._lmdb_paths)

    def _get_env(self, fi: int) -> lmdb.Environment:
        env = self._envs[fi]
        if env is None:
            env = lmdb.open(
                self._lmdb_paths[fi],
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
            self._envs[fi] = env
        return env

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int):
        fi, key = self._keys[idx]
        env = self._get_env(fi)
        with env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(f"Missing LMDB value: file={self._lmdb_paths[fi]}, key={key!r}")
        item = pickle.loads(raw)
        if not isinstance(item, dict):
            raise TypeError(f"LMDB value is not a dict: file={self._lmdb_paths[fi]}, key={key!r}")
        if "atoms" not in item or "coordinates" not in item:
            raise KeyError(f"Sample missing atoms/coordinates: file={self._lmdb_paths[fi]}, key={key!r}")

        atoms = item["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        # Map atomic numbers (ints) to element symbols (str)
        atoms = atoms_to_symbols(atoms)

        coords = np.asarray(item["coordinates"], dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coordinates must be (N,3) or (1,N,3); got shape={coords.shape}")
        if len(atoms) != coords.shape[0]:
            raise ValueError(f"len(atoms)={len(atoms)} != coords rows={coords.shape[0]}")

        atoms, coords = preprocess_molecule(
            atoms,
            coords,
            remove_h=self.remove_hydrogen,
            center=self.center_coords,
            max_atoms=self.max_atoms,
        )
        if atoms is None:  # filtered by preprocess_molecule
            return None

        # Cap atom count after dehydrogenation
        if len(atoms) > self.max_atoms:
            atoms = atoms[: self.max_atoms]
            coords = coords[: self.max_atoms]
        return {"atoms": atoms, "coordinates": coords}

    def close(self):
        for env in self._envs:
            if env is not None:
                env.close()


class UniMolDataCollator:
    """Collate dict batches from the dataset into model tensors."""

    def __init__(
        self,
        dictionary: Any,
        mask_idx: int,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        mask_prob: float = 0.15,
        noise_type: str = "uniform",
        noise: float = 1.0,
        seed: int = 42,
    ):
        self.dictionary = dictionary
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.mask_prob = mask_prob
        self.noise_type = noise_type
        self.noise = noise
        self.seed = seed

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Drop None entries (filtered samples)
        features = [f for f in features if f is not None]
        if not features:
            # Empty batch; Trainer handles this
            return {}

        list_atoms = [f["atoms"] for f in features]
        list_coordinates = [f["coordinates"] for f in features]

        # Max sequence length in this batch
        batch_seq_lens = [1 + len(atoms) + 1 for atoms in list_atoms]  # BOS + atoms + EOS
        L = max(batch_seq_lens)

        num_types = len(self.dictionary)
        # Time-based seed so each collate call gets fresh masking noise
        import time
        current_seed = self.seed + int(time.time() * 1000) % 10000
        rng = np.random.default_rng(current_seed)

        list_src_tokens = []
        list_tokens_target = []
        list_src_coord = []
        list_coord_target = []
        list_src_distance = []
        list_distance_target = []
        list_src_edge_type = []

        for atoms, coordinates in zip(list_atoms, list_coordinates):
            n_atoms = len(atoms)
            assert n_atoms == coordinates.shape[0]

            # 1. Tokenize atoms
            token_ids = [self.dictionary.index(a) for a in atoms]
            seq = [self.bos_idx] + token_ids + [self.eos_idx]
            pad_len = max(0, L - len(seq))
            src_tokens = torch.tensor([seq], dtype=torch.long)
            src_tokens = torch.nn.functional.pad(src_tokens, (0, pad_len), value=self.pad_idx)

            # 2. Mask tokens
            tokens_target = torch.full_like(src_tokens, self.pad_idx)
            num_mask = max(1, int(self.mask_prob * n_atoms) + (1 if rng.random() < 0.5 else 0))
            mask_positions = rng.choice(n_atoms, min(num_mask, n_atoms), replace=False)
            mask_indices = mask_positions + 1  # +1 for BOS at index 0

            for idx in mask_indices:
                tokens_target[0, idx] = src_tokens[0, idx].item()
                src_tokens[0, idx] = self.mask_idx

            # 3. Coordinates
            coords = np.asarray(coordinates, dtype=np.float32)
            coord_bos = np.zeros((1, 3), dtype=np.float32)
            coord_eos = np.zeros((1, 3), dtype=np.float32)
            coord_pad = np.zeros((pad_len, 3), dtype=np.float32)
            full_coord = np.vstack([coord_bos, coords, coord_eos, coord_pad])
            src_coord = torch.from_numpy(full_coord).unsqueeze(0)
            coord_target = src_coord.clone()

            # 4. Coordinate noise on masked atom positions
            if self.noise_type is not None and self.noise_type.lower() != "none" and len(mask_indices) > 0:
                num_mask = len(mask_indices)
                if self.noise_type == "trunc_normal":
                    noise_arr = np.clip(
                        rng.standard_normal((num_mask, 3)) * self.noise,
                        a_min=-self.noise * 2.0,
                        a_max=self.noise * 2.0,
                    ).astype(np.float32)
                elif self.noise_type == "normal":
                    noise_arr = (rng.standard_normal((num_mask, 3)) * self.noise).astype(np.float32)
                elif self.noise_type == "uniform":
                    noise_arr = rng.uniform(low=-self.noise, high=self.noise, size=(num_mask, 3)).astype(np.float32)
                else:
                    noise_arr = np.zeros((num_mask, 3), dtype=np.float32)

                coord_np = src_coord[0].cpu().numpy()
                coord_np[mask_indices, :] += noise_arr
                src_coord = torch.from_numpy(coord_np).unsqueeze(0)

            # 5. Distance matrix
            dist = distance_matrix(full_coord, full_coord).astype(np.float32)
            src_distance = torch.from_numpy(dist).unsqueeze(0)
            distance_target = src_distance.clone()

            # 6. Edge type
            tokens_np = src_tokens[0].numpy()
            ei = tokens_np.reshape(-1, 1)
            ej = tokens_np.reshape(1, -1)
            offset = ei * num_types + ej
            src_edge_type = torch.from_numpy(offset.astype(np.int64)).unsqueeze(0)

            list_src_tokens.append(src_tokens)
            list_tokens_target.append(tokens_target)
            list_src_coord.append(src_coord)
            list_coord_target.append(coord_target)
            list_src_distance.append(src_distance)
            list_distance_target.append(distance_target)
            list_src_edge_type.append(src_edge_type)

        return {
            "src_tokens": torch.cat(list_src_tokens, dim=0),
            "tokens_target": torch.cat(list_tokens_target, dim=0),
            "src_coord": torch.cat(list_src_coord, dim=0),
            "coord_target": torch.cat(list_coord_target, dim=0),
            "src_distance": torch.cat(list_src_distance, dim=0),
            "distance_target": torch.cat(list_distance_target, dim=0),
            "src_edge_type": torch.cat(list_src_edge_type, dim=0),
            # Dummy labels so Trainer runs loss / eval paths
            "labels": torch.zeros(len(list_atoms), dtype=torch.long),  # unused placeholder
        }


class UniMolTrainer(Trainer):
    """Trainer with Uni-Mol forward signature and combined pretraining loss."""

    def __init__(self, model_args, dist_mean, dist_std, step_offset=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.dist_mean = dist_mean
        self.dist_std = dist_std
        self.pad_idx = model_args.pad_idx
        self.step_offset = step_offset  # offset when resuming (checkpoint step in folder name)
        # label_names: required so prediction_step computes eval_loss when labels are present
        self.label_names = ["labels"]

    def _save_checkpoint(self, model, trial, **kwargs):
        """Add step_offset to global_step during save so folder names stay continuous."""
        if self.step_offset > 0:
            self.state.global_step += self.step_offset
            try:
                super()._save_checkpoint(model, trial, **kwargs)
            finally:
                self.state.global_step -= self.step_offset
        else:
            super()._save_checkpoint(model, trial, **kwargs)

    def create_optimizer(self):
        """
        Build AdamW with betas/eps matching the original Uni-Mol pretrain recipe.
        DeepSpeed calls this when no optimizer is defined in the DS config.
        """
        if self.optimizer is None:
            if hasattr(self, 'get_parameter_groups'):
                optimizer_grouped_parameters = self.get_parameter_groups()
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if p.requires_grad],
                        "weight_decay": self.args.weight_decay,
                    }
                ]
            
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(0.9, 0.99),
                eps=1e-6,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Uni-Mol masked token / coord / distance (+ optional norm) losses."""
        # Unpack inputs
        src_tokens = inputs["src_tokens"]
        src_distance = inputs["src_distance"]
        src_coord = inputs["src_coord"]
        src_edge_type = inputs["src_edge_type"]
        tokens_target = inputs["tokens_target"]
        coord_target = inputs["coord_target"]
        distance_target = inputs["distance_target"]

        masked_tokens = tokens_target.ne(self.pad_idx)

        # Forward
        outputs = model(
            src_tokens=src_tokens,
            src_distance=src_distance,
            src_coord=src_coord,
            src_edge_type=src_edge_type,
            encoder_masked_tokens=masked_tokens,
        )
        
        # Uni-Mol returns a tuple
        if isinstance(outputs, tuple) and len(outputs) >= 5:
            logits_encoder, encoder_distance, encoder_coord, x_norm, delta_encoder_pair_rep_norm = outputs
        else:
            # Fallback unpack
            logits_encoder = outputs[0] if isinstance(outputs, tuple) else outputs
            encoder_distance = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
            encoder_coord = outputs[2] if isinstance(outputs, tuple) and len(outputs) > 2 else None
            x_norm = outputs[3] if isinstance(outputs, tuple) and len(outputs) > 3 else None
            delta_encoder_pair_rep_norm = outputs[4] if isinstance(outputs, tuple) and len(outputs) > 4 else None

        sample_size = masked_tokens.long().sum()
        if sample_size == 0:
            loss = torch.tensor(0.0, device=src_tokens.device)
            log_out = {"loss": 0.0}
            if return_outputs:
                # Dict shape for prediction_step / logits
                model_outputs = {
                    "logits": logits_encoder if 'logits_encoder' in locals() else None,
                }
                return (loss, model_outputs)
            return loss
        else:
            # Masked token loss
            target = tokens_target[masked_tokens]
            masked_token_loss = F.nll_loss(
                F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
                target,
                ignore_index=self.pad_idx,
                reduction="mean",
            )
            loss = masked_token_loss * self.model_args.masked_token_loss
            log_out = {"masked_token_loss": masked_token_loss.item()}

            # Masked coord loss
            if encoder_coord is not None:
                masked_coord_loss = F.smooth_l1_loss(
                    encoder_coord[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                    beta=1.0,
                )
                loss = loss + masked_coord_loss * self.model_args.masked_coord_loss
                log_out["masked_coord_loss"] = masked_coord_loss.item()

            # Masked distance loss
            if encoder_distance is not None:
                masked_distance = encoder_distance[masked_tokens, :]
                masked_distance_target = distance_target[masked_tokens]
                masked_src = src_tokens.ne(self.pad_idx)
                nb = masked_tokens.sum(dim=-1)
                masked_src_expanded = torch.repeat_interleave(masked_src.float(), nb.long(), dim=0)
                dt = (masked_distance_target.float() - self.dist_mean) / self.dist_std
                masked_dist_loss = F.smooth_l1_loss(
                    masked_distance[masked_src_expanded.bool()].view(-1).float(),
                    dt[masked_src_expanded.bool()].view(-1),
                    reduction="mean",
                    beta=1.0,
                )
                loss = loss + masked_dist_loss * self.model_args.masked_dist_loss
                log_out["masked_dist_loss"] = masked_dist_loss.item()

            # X norm loss
            if getattr(self.model_args, "x_norm_loss", 0) > 0 and x_norm is not None:
                loss = loss + self.model_args.x_norm_loss * x_norm
                log_out["x_norm_loss"] = x_norm.item()

            # Delta pair repr norm loss
            if getattr(self.model_args, "delta_pair_repr_norm_loss", 0) > 0 and delta_encoder_pair_rep_norm is not None:
                loss = loss + self.model_args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
                log_out["delta_pair_repr_norm_loss"] = delta_encoder_pair_rep_norm.item()

            log_out["loss"] = loss.item()

        if return_outputs:
            model_outputs = {
                "logits": logits_encoder,
            }
            # Extra tensors for metrics hooks if needed
            if encoder_distance is not None:
                model_outputs["encoder_distance"] = encoder_distance
            if encoder_coord is not None:
                model_outputs["encoder_coord"] = encoder_coord
            return (loss, model_outputs)
        else:
            return loss


class WandbLogPrintCallback(TrainerCallback):
    """
    Print metrics on each log step so wandb's Logs tab captures stdout reliably.
    Prefix with [Step N] so identical-looking lines are not collapsed.
    """

    def __init__(self, step_offset=0):
        self.step_offset = step_offset

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            log_str = {k: f"{v:.4g}" if isinstance(v, float) else str(v) for k, v in logs.items()}
            step = getattr(state, "global_step", 0) + self.step_offset
            print(f"[Step {step}] {log_str}", flush=True)
            sys.stdout.flush()

def set_seed(seed: int):
    """Set Python, NumPy, and PyTorch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):

    set_seed(args.seed)

    # Dictionary
    from unicore.data import Dictionary

    dict_path = args.dict
    if not os.path.isfile(dict_path):
        raise FileNotFoundError(f"dict not found: {dict_path}")
    dictionary = Dictionary.load(dict_path)
    mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
    pad_idx = dictionary.pad()
    bos_idx = dictionary.bos()
    eos_idx = dictionary.eos()

    print(f"Dictionary: {len(dictionary)} types, pad={pad_idx}, bos={bos_idx}, eos={eos_idx}, mask={mask_idx}")

    # Model hyperparameters (Namespace for UniMolModel)
    from argparse import Namespace

    model_args = Namespace(
        encoder_layers=15,
        encoder_embed_dim=512,
        encoder_ffn_embed_dim=2048,
        encoder_attention_heads=64,
        emb_dropout=0.1,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        pooler_dropout=0.0,
        max_seq_len=512,
        activation_fn="gelu",
        pooler_activation_fn="tanh",
        post_ln=False,
        masked_token_loss=1.0,
        masked_coord_loss=5.0,
        masked_dist_loss=10.0,
        x_norm_loss=0.01,
        delta_pair_repr_norm_loss=0.01,
        mode="train",
        mask_prob=0.15,
        noise_type="uniform",
        noise=1.0,
        seed=args.seed,
        pad_idx=pad_idx,
    )

    # Model
    from unimol import UniMolModel

    model = UniMolModel(model_args, dictionary)

    # Datasets
    train_ds = LMDBPretrainDataset(
        args.train_path,
        remove_hydrogen=True,
        center_coords=True,
        max_atoms=args.max_atoms,
    )
    valid_ds = LMDBPretrainDataset(
        args.valid_path,
        remove_hydrogen=True,
        center_coords=True,
        max_atoms=args.max_atoms,
    )

    # Optional subsample for smoke tests
    if args.sample_size is not None and args.sample_size > 0:
        print(f"Limiting datasets to {args.sample_size} samples for testing")
        train_ds_size = len(train_ds)
        valid_ds_size = len(valid_ds)
        train_indices = list(range(min(args.sample_size, train_ds_size)))
        valid_indices = list(range(min(args.sample_size, valid_ds_size)))
        train_ds = Subset(train_ds, train_indices)
        valid_ds = Subset(valid_ds, valid_indices)
        print(f"  Train: {train_ds_size} -> {len(train_ds)} samples")
        print(f"  Valid: {valid_ds_size} -> {len(valid_ds)} samples")

    # Collator
    data_collator = UniMolDataCollator(
        dictionary=dictionary,
        mask_idx=mask_idx,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        mask_prob=model_args.mask_prob,
        noise_type=model_args.noise_type,
        noise=model_args.noise,
        seed=model_args.seed,
    )

    # TrainingArguments (Adam betas/eps also set for non-DeepSpeed; DS may override via JSON)
    # Resume?
    resume_ckpt = args.resume_from_checkpoint
    is_resume = resume_ckpt and resume_ckpt.lower() not in ("none", "")

    # step_offset from folder name, e.g. checkpoint-500000 -> 500000
    step_offset = 0
    if is_resume:
        import re
        match = re.search(r"checkpoint-(\d+)", resume_ckpt)
        if match:
            step_offset = int(match.group(1))
            print(f"  Parsed step_offset={step_offset} from checkpoint path: {resume_ckpt}")

    # On resume: lower LR and separate warmup
    effective_lr = args.resume_lr if is_resume else args.lr
    effective_warmup = args.resume_warmup_steps if is_resume else args.warmup_steps

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=effective_lr,
        weight_decay=args.weight_decay,
        warmup_steps=effective_warmup,
        lr_scheduler_type="polynomial",
        logging_steps=args.log_interval,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        fp16=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=(
            args.wandb_name
            if args.wandb_name
            else f"pretrain-{os.path.basename(args.output_dir)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(1000, 9999)}"
        ),
        deepspeed=args.deepspeed,
        # Match original recipe: betas=(0.9, 0.99), eps=1e-6
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=1e-6,
    )

    # wandb
    wandb.init(
        project=args.wandb_project,
        name=training_args.run_name,
        config=vars(args),
    )

    # Trainer + stdout callback for wandb Logs
    trainer = UniMolTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        model_args=model_args,
        dist_mean=DIST_MEAN,
        dist_std=DIST_STD,
        step_offset=step_offset,
        callbacks=[WandbLogPrintCallback(step_offset=step_offset)],
    )

    # Train
    if is_resume:
        if not os.path.isdir(resume_ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")

        # Resume strategy: load model + optimizer (Adam moments) via Trainer,
        # fresh LR scheduler from TrainingArguments; global_step temporarily zeroed below.
        import json as json_mod
        import shutil

        # 1. Patch deepspeed_load_checkpoint: load optimizer, skip scheduler state
        _ds_patched = False
        _orig_ds_load = None
        try:
            import transformers.trainer as _trainer_module
            _orig_ds_load = getattr(_trainer_module, 'deepspeed_load_checkpoint', None)
            if _orig_ds_load is not None:
                import glob as _glob_mod
                def _patched_ds_load(deepspeed_engine, checkpoint_path, load_module_strict=True):
                    ds_dirs = sorted(_glob_mod.glob(f"{checkpoint_path}/global_step*"))
                    if ds_dirs:
                        print("  [Patched] Loading DeepSpeed checkpoint (optimizer=yes, scheduler=no)")
                        load_path, _ = deepspeed_engine.load_checkpoint(
                            checkpoint_path,
                            load_module_strict=load_module_strict,
                            load_optimizer_states=True,
                            load_lr_scheduler_states=False,
                        )
                        if load_path is None:
                            raise ValueError(f"[deepspeed] failed to resume from {checkpoint_path}")
                    else:
                        raise ValueError(f"No DeepSpeed checkpoint at {checkpoint_path}")
                _trainer_module.deepspeed_load_checkpoint = _patched_ds_load
                _ds_patched = True
                print("  Patched: load optimizer states (Adam m/v), not scheduler")
        except Exception as e:
            print(f"  Warning: Could not patch deepspeed_load_checkpoint: {e}")

        # 2. Temporarily reset global_step in trainer_state.json so Trainer does not skip data
        state_file = os.path.join(resume_ckpt, "trainer_state.json")
        state_backup = state_file + ".resume_bak"
        _state_modified = False
        if os.path.isfile(state_file):
            shutil.copy2(state_file, state_backup)
            with open(state_file, 'r') as f:
                saved_state = json_mod.load(f)
            saved_state['global_step'] = 0
            saved_state['epoch'] = 0.0
            saved_state['best_metric'] = None
            saved_state['best_model_checkpoint'] = None
            saved_state['log_history'] = []
            with open(state_file, 'w') as f:
                json_mod.dump(saved_state, f, indent=2)
            _state_modified = True
            print("  Temporarily set global_step=0 in trainer_state.json")

        print(f"\n===== Resuming training (model + optimizer, fresh scheduler) =====")
        print(f"  checkpoint:     {resume_ckpt}")
        print(f"  learning_rate:  {effective_lr}")
        print(f"  warmup_steps:   {effective_warmup}")
        print(f"  max_steps:      {args.max_steps}")
        print(f"  step_offset:    {step_offset}")
        print("  Policy: load weights + Adam moments, new scheduler, counter starts at step 0")

        try:
            trainer.train(resume_from_checkpoint=resume_ckpt)
        finally:
            # Restore trainer_state.json
            if _state_modified and os.path.isfile(state_backup):
                shutil.move(state_backup, state_file)
            # Restore deepspeed_load_checkpoint
            if _ds_patched and _orig_ds_load is not None:
                try:
                    _trainer_module.deepspeed_load_checkpoint = _orig_ds_load
                except Exception:
                    pass
    else:
        print(f"\n===== Starting training from scratch: max_steps={args.max_steps} =====")
        trainer.train()

    # Best checkpoint summary
    print(f"\n{'='*60}")
    print(f"Training Summary:")
    print(f"{'='*60}")
    
    if trainer.state.best_model_checkpoint is not None:
        print("\nBest model checkpoint:")
        print(f"  - Checkpoint path: {trainer.state.best_model_checkpoint}")
        print(f"  - Best metric ({trainer.args.metric_for_best_model}): {trainer.state.best_metric:.6f}")
        print(f"  - Best global step: {trainer.state.best_global_step}")
        print(f"\n  Note: The best model has been automatically loaded into trainer.model")
        print(f"        You can use it directly or load from: {trainer.state.best_model_checkpoint}")
    else:
        print("\nWarning: No best model checkpoint found.")
        print("  This might happen if:")
        print("    - No evaluation was performed during training")
        print("    - All checkpoints had the same metric value")
    
    # List saved checkpoints
    import json
    import glob
    from pathlib import Path
    
    output_dir = Path(args.output_dir)
    checkpoint_dirs = sorted(
        [d for d in output_dir.glob(f"{PREFIX_CHECKPOINT_DIR}-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0
    )
    
    if checkpoint_dirs:
        print(f"\n{'='*60}")
        print(f"All Saved Checkpoints:")
        print(f"{'='*60}")
        print(f"{'Step':<10} {'Checkpoint Path':<50} {'Is Best':<10}")
        print("-" * 70)
        
        for ckpt_dir in checkpoint_dirs:
            step = ckpt_dir.name.split("-")[-1]
            is_best = "yes" if (trainer.state.best_model_checkpoint and 
                             str(ckpt_dir) == trainer.state.best_model_checkpoint) else ""
            print(f"{step:<10} {str(ckpt_dir):<50} {is_best:<10}")
        
        # eval_loss history from trainer_state.json
        trainer_state_file = output_dir / "trainer_state.json"
        if trainer_state_file.exists():
            try:
                with open(trainer_state_file, 'r') as f:
                    trainer_state = json.load(f)
                
                log_history = trainer_state.get("log_history", [])
                eval_losses = {log.get("step", -1): log.get("eval_loss") 
                              for log in log_history if "eval_loss" in log}
                
                if eval_losses:
                    print(f"\n{'='*60}")
                    print(f"Checkpoint Evaluation Metrics:")
                    print(f"{'='*60}")
                    print(f"{'Step':<10} {'Eval Loss':<15} {'Is Best':<10}")
                    print("-" * 35)
                    for step, eval_loss in sorted(eval_losses.items()):
                        is_best = "yes" if (trainer.state.best_global_step and 
                                         step == trainer.state.best_global_step) else ""
                        print(f"{step:<10} {eval_loss:<15.6f} {is_best:<10}")
            except Exception as e:
                print(f"\nCould not read evaluation metrics from trainer_state.json: {e}")
    
    print(f"{'='*60}\n")

    # Optional test eval (trainer.model is best eval_loss checkpoint if load_best_model_at_end)
    if args.test_path and os.path.exists(args.test_path):
        print("\n" + "="*60)
        print("Evaluating on test set using BEST model...")
        print("="*60)
        
        # Confirm which checkpoint is loaded
        if trainer.state.best_model_checkpoint is not None:
            print("Using best model:")
            print(f"  - Step: {trainer.state.best_global_step}")
            print(f"  - Checkpoint: {trainer.state.best_model_checkpoint}")
            print(f"  - Best eval_loss: {trainer.state.best_metric:.6f}")
            print(f"\n  Note: Best model has been automatically loaded by Trainer")
            print(f"        (load_best_model_at_end=True)")
        else:
            print("Warning: No best model checkpoint found.")
            print("  Using the last checkpoint for test evaluation.")
        
        test_ds = LMDBPretrainDataset(
            args.test_path,
            remove_hydrogen=True,
            center_coords=True,
            max_atoms=args.max_atoms,
        )
        if args.sample_size is not None and args.sample_size > 0:
            test_ds_size = len(test_ds)
            test_indices = list(range(min(args.sample_size, test_ds_size)))
            test_ds = Subset(test_ds, test_indices)
            print(f"\n  Test dataset: {test_ds_size} -> {len(test_ds)} samples")

        print("\nRunning evaluation...")
        test_results = trainer.evaluate(test_ds)
        print(f"\n{'='*60}")
        print(f"Test Results (using best model from step {trainer.state.best_global_step if trainer.state.best_global_step else 'N/A'}):")
        print(f"{'='*60}")
        for key, value in test_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*60}\n")

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Uni-Mol pretraining (Transformers Trainer + DeepSpeed)")
    parser.add_argument("--dict", type=str, default="/data/Jingyuan_data/OMol25_MC_all/dict.txt", help="Path to dict.txt")
    parser.add_argument("--train-path", type=str, default="/data/Jingyuan_data/OMol25_MC_all/train", help="Train LMDB file or directory")
    parser.add_argument("--valid-path", type=str, default="/data/Jingyuan_data/OMol25_MC_all/valid.lmdb", help="Validation LMDB path")
    parser.add_argument("--test-path", type=str, default="/data/Jingyuan_data/OMol25_MC_all/test.lmdb", help="Test LMDB path")
    parser.add_argument("--output-dir", type=str, default="/data/Jingyuan_data/Uni-Mol", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=500000, help="Max training steps")
    parser.add_argument("--save-steps", type=int, default=10000, help="Checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=10000, help="Evaluate every N steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=5000, help="LR warmup steps (matches README warmup-updates)")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    parser.add_argument("--max-atoms", type=int, default=256, help="Max atoms after removing hydrogens")
    parser.add_argument("--sample-size", type=int, default=None, help="Cap train/valid/test size for quick tests")
    parser.add_argument("--deepspeed", type=str, default='/home/zhujingyuan/simple-uni-mol/deepspeed_config.json', help="DeepSpeed JSON config path")
    parser.add_argument("--wandb-project", type=str, default="unimol-pretrain", help="wandb project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader num_workers")
    parser.add_argument("--log-interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--resume-from-checkpoint", type=str, default="/data/Jingyuan_data/Uni-Mol/checkpoint-500000",
                        help="Resume from this checkpoint; use None or empty string to train from scratch.")
    parser.add_argument("--resume-lr", type=float, default=1e-5,
                        help="LR when resuming (default 1e-5, lower than initial 1e-4 to reduce oscillation)")
    parser.add_argument("--resume-warmup-steps", type=int, default=2000,
                        help="Warmup steps after resume (ramps LR from 0 to resume-lr; default 2000)")
    # DeepSpeed injects --local_rank; keep this argument even though Trainer handles distribution.
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (passed by DeepSpeed / launcher)")

    args = parser.parse_args()
    main(args)
