"""
Evaluate Property-3D-only (single-token, no SMILES) checkpoint on test LMDB (MAE / R²).

Example:
  CUDA_VISIBLE_DEVICES=0 python -u ablation_experiments/inference_property_3d_only.py \\
    --Stage3_ckpt /data/jingyuan_data/Stage3_Property_dipole_moment_3d_only_ckpt/checkpoint-3000 \\
    --test_lmdb /data/jingyuan_data/tmqmg/stage3/test/tmqmg_atom_only_new.lmdb \\
    --property dipole_moment
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from typing import Optional

import lmdb
import numpy as np
import torch
from transformers import AutoTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_STAGE3_DIR = os.path.join(_PROJECT_ROOT, "Stage3")
if _STAGE3_DIR not in sys.path:
    sys.path.insert(0, _STAGE3_DIR)

import utils  # noqa: F401
from Property import PROPERTY_CONFIG
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel, generate_with_single_token_structure
from train_defaults import VASKA_DEFAULTS
from utils import _atoms_coords_remove_h_center, _lmdb_env_kwargs

from ablation_experiments.dataset_3d_only import INSTRUCTION_3D_ONLY, TmQMg3DOnlyUnimolDataset

DEFAULT_TEST_LMDB = "/data/jingyuan_data/tmqmg/stage3/test/tmqmg_atom_only_new.lmdb"


def _parse_first_float(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _collect_eval_samples(dataset: TmQMg3DOnlyUnimolDataset):
    samples = []
    for lmdb_path, key_bytes in dataset.key_index:
        env = dataset._envs.get(lmdb_path)
        if env is None:
            env = lmdb.open(lmdb_path, **_lmdb_env_kwargs())
            dataset._envs[lmdb_path] = env
        with env.begin() as txn:
            data = pickle.loads(txn.get(key_bytes))
        y = float(data[dataset.property_key])
        atoms = data["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(data["coordinates"], dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
        samples.append((atoms, coords, dataset.instruction.strip(), y))
    return samples


def main():
    p = argparse.ArgumentParser(description="Property-3D-only test eval (MAE/R²).")
    p.add_argument("--model_name", type=str, default=VASKA_DEFAULTS["model_name"])
    p.add_argument("--3D_encoder_dict", dest="three_d_encoder_dict", type=str, default=VASKA_DEFAULTS["3D_encoder_dict"])
    p.add_argument("--Stage3_ckpt", dest="stage3_ckpt", type=str, required=True)
    p.add_argument("--test_lmdb", type=str, default=DEFAULT_TEST_LMDB)
    p.add_argument("--property", type=str, default="dipole_moment", choices=list(PROPERTY_CONFIG.keys()))
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_target", type=str, default="all", choices=["qv", "qkv", "all"])
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--print_every", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--save_json", type=str, default=None)
    args = p.parse_args()

    if not os.path.isfile(args.test_lmdb):
        raise FileNotFoundError(f"Test LMDB not found: {args.test_lmdb}")
    if not os.path.isdir(args.stage3_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.stage3_ckpt}")

    prop_cfg = PROPERTY_CONFIG[args.property]
    prop_key = prop_cfg["key"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalModel(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        init_ckpt=args.stage3_ckpt,
        train_3d_encoder=False,
        train_projection=False,
        train_lora=False,
        load_pretrained_projection=True,
        load_pretrained_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
    )
    model.eval()

    dataset = TmQMg3DOnlyUnimolDataset(
        [args.test_lmdb],
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        property_key=prop_key,
        instruction=INSTRUCTION_3D_ONLY[prop_key],
    )
    test_data = _collect_eval_samples(dataset)
    print(
        f"[Eval-3D-only] property={args.property} | test={len(test_data)} | "
        f"prompt=no SMILES | ckpt={args.stage3_ckpt}"
    )

    y_true, y_pred = [], []
    n_parse_fail = 0
    pred_records = []

    for i, (atoms, coords, user_content, y) in enumerate(test_data, start=1):
        out = generate_with_single_token_structure(
            model,
            tokenizer,
            atoms,
            coords,
            user_content,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        pred = _parse_first_float(out)
        if pred is None or not np.isfinite(pred):
            n_parse_fail += 1
            pred_records.append({"ref": float(y), "pred_text": out, "pred_value": None})
        else:
            y_true.append(y)
            y_pred.append(pred)
            pred_records.append({"ref": float(y), "pred_text": out, "pred_value": float(pred)})

        if args.print_every > 0 and i % args.print_every == 0:
            print(f"[Eval-3D-only] {i}/{len(test_data)} | parsed={len(y_pred)} | parse_fail={n_parse_fail}")

    if len(y_pred) == 0:
        raise RuntimeError("No valid numeric predictions.")

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    mae = _mae(y_true_arr, y_pred_arr)
    r2 = _r2(y_true_arr, y_pred_arr)

    print("\n================ Property-3D-only Test Metrics ================")
    print(f"property:       {args.property} ({prop_cfg.get('unit', '')})")
    print(f"test_total:     {len(test_data)}")
    print(f"pred_parsed:    {len(y_pred_arr)}")
    print(f"parse_fail:     {n_parse_fail}")
    print(f"MAE:            {mae:.6f}")
    print(f"R2:             {r2:.6f}" if np.isfinite(r2) else "R2:             nan")
    print("===============================================================")

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "property": args.property,
                    "ablation": "3d_only_single_token",
                    "ckpt": args.stage3_ckpt,
                    "test_lmdb": args.test_lmdb,
                    "mae": mae,
                    "r2": r2 if np.isfinite(r2) else None,
                    "predictions": pred_records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[Eval-3D-only] Saved: {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
