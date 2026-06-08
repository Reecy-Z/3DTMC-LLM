"""
Evaluate NiComplex OOD checkpoint on the held-out test split (MAE / R²).

Example:
  CUDA_VISIBLE_DEVICES=0 python -u OOD/NiComplex/inference_nicomplex_ood.py \\
    --experiment train_rest_test_Pybox \\
    --Stage3_ckpt /data/jingyuan_data/NiComplex_OOD_Models/exp_train_rest_test_Pybox
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_STAGE3_DIR = os.path.join(_PROJECT_ROOT, "Stage3")
if _STAGE3_DIR not in sys.path:
    sys.path.insert(0, _STAGE3_DIR)

import utils  # noqa: F401
from NiComplex import NI_INSTRUCTION, load_merged_valid_nicomplex_records
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel, generate_with_single_token_structure
from train_defaults import NICOMPLEX_DEFAULTS, VASKA_DEFAULTS
from utils import _atoms_coords_remove_h_center, format_instruction_field

from OOD.NiComplex.NiComplex_OOD import DEFAULT_LMDB, DEFAULT_OUTPUT_DIR
from OOD.NiComplex.nicomplex_split import (
    EXPERIMENT_NAMES,
    experiment_dirname,
    split_by_experiment,
    summarize_field,
)


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


def _build_user_content(sample: dict) -> str:
    smiles = format_instruction_field(sample.get("smiles", ""))
    temp = format_instruction_field(sample.get("temp", ""))
    if "{smiles}" in NI_INSTRUCTION or "{temp}" in NI_INSTRUCTION:
        return NI_INSTRUCTION.format(smiles=smiles, temp=temp)
    return NI_INSTRUCTION


def _collect_eval_samples(ood_samples: list) -> list:
    rows = []
    for sample in ood_samples:
        y = float(sample["ddG"])
        atoms = sample["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(sample["coordinates"], dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
        rows.append((atoms, coords, _build_user_content(sample), y, sample.get("R_Type"), sample.get("L_Scaffold")))
    return rows


def _eval_one_experiment(args, experiment_name: str, all_valid: list) -> dict:
    ckpt = args.stage3_ckpt
    if not ckpt:
        ckpt = os.path.join(args.output_dir, experiment_dirname(experiment_name))
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(f"Checkpoint not found for {experiment_name}: {ckpt}")

    _, ood_samples, spec = split_by_experiment(all_valid, experiment_name)
    test_data = _collect_eval_samples(ood_samples)
    if not test_data:
        raise RuntimeError(f"No valid OOD test samples for experiment={experiment_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalModel(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        init_ckpt=ckpt,
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

    print(
        f"[Eval-NiComplex-OOD] experiment={experiment_name} | split_col={spec['split_col']} | "
        f"ood_test={len(test_data)} | ckpt={ckpt}"
    )

    y_true, y_pred = [], []
    n_parse_fail = 0
    pred_records = []

    for i, (atoms, coords, user_content, y, r_type, scaffold) in enumerate(test_data, start=1):
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
            pred_records.append(
                {
                    "ref": float(y),
                    "R_Type": r_type,
                    "L_Scaffold": scaffold,
                    "pred_text": out,
                    "pred_value": None,
                }
            )
        else:
            y_true.append(y)
            y_pred.append(pred)
            pred_records.append(
                {
                    "ref": float(y),
                    "R_Type": r_type,
                    "L_Scaffold": scaffold,
                    "pred_text": out,
                    "pred_value": float(pred),
                }
            )

        if args.print_every > 0 and i % args.print_every == 0:
            print(f"[Eval-NiComplex-OOD] {i}/{len(test_data)} | parsed={len(y_pred)} | parse_fail={n_parse_fail}")

    if not y_pred:
        raise RuntimeError("No valid numeric predictions.")

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    mae = _mae(y_true_arr, y_pred_arr)
    r2 = _r2(y_true_arr, y_pred_arr)

    print(f"\n--- OOD experiment={experiment_name} ---")
    print(f"MAE: {mae:.6f} kcal/mol | R2: {r2:.6f} | N={len(y_pred_arr)} | parse_fail={n_parse_fail}")

    result = {
        "experiment": experiment_name,
        "split_col": spec["split_col"],
        "train_types": sorted(spec["train_types"]),
        "test_types": sorted(spec["test_types"]),
        "ckpt": ckpt,
        "n_ood_test": len(test_data),
        "n_parsed": len(y_pred_arr),
        "n_parse_fail": n_parse_fail,
        "mae": mae,
        "r2": r2 if np.isfinite(r2) else None,
        "predictions": pred_records,
    }

    if args.save_json or args.run_all_experiments:
        if args.run_all_experiments:
            out_path = os.path.join(args.output_dir, experiment_dirname(experiment_name), "ood_test_predictions.json")
        else:
            out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Eval-NiComplex-OOD] Saved {out_path}")

    return result


def main():
    p = argparse.ArgumentParser(description="NiComplex OOD inference")
    p.add_argument("--model_name", type=str, default=VASKA_DEFAULTS["model_name"])
    p.add_argument("--3D_encoder_dict", dest="three_d_encoder_dict", type=str, default=VASKA_DEFAULTS["3D_encoder_dict"])
    p.add_argument("--Stage3_ckpt", dest="stage3_ckpt", type=str, default=None)
    p.add_argument("--lmdb", action="append", default=None, dest="lmdb_paths", metavar="PATH")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--experiment", type=str, default=None, choices=list(EXPERIMENT_NAMES))
    p.add_argument("--run_all_experiments", action="store_true")
    p.add_argument("--lora_r", type=int, default=NICOMPLEX_DEFAULTS["lora_r"])
    p.add_argument("--lora_alpha", type=int, default=NICOMPLEX_DEFAULTS["lora_alpha"])
    p.add_argument("--lora_target", type=str, default=NICOMPLEX_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"])
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--save_json", type=str, default=None)
    args = p.parse_args()

    if not args.run_all_experiments and not args.experiment:
        p.error("Specify --experiment or --run_all_experiments")

    lmdb_paths = args.lmdb_paths or [DEFAULT_LMDB]
    print(f"[Eval-NiComplex-OOD] Reading LMDB: {lmdb_paths}")
    all_valid, _ = load_merged_valid_nicomplex_records(lmdb_paths, local_rank=0)
    print(
        f"[Eval-NiComplex-OOD] loaded {len(all_valid)} samples | "
        f"R_Type={summarize_field(all_valid, 'R_Type')} | "
        f"L_Scaffold={summarize_field(all_valid, 'L_Scaffold')}"
    )

    experiments = list(EXPERIMENT_NAMES) if args.run_all_experiments else [args.experiment]
    summary = []
    for experiment_name in experiments:
        summary.append(_eval_one_experiment(args, experiment_name, all_valid))

    if len(summary) > 1:
        print("\n================ NiComplex OOD summary ================")
        for row in summary:
            r2s = "nan" if row["r2"] is None else f"{row['r2']:.4f}"
            print(f"  {row['experiment']:30s}  MAE={row['mae']:.4f}  R2={r2s}  N={row['n_parsed']}")
        maes = [r["mae"] for r in summary]
        print(f"  mean MAE across experiments: {float(np.mean(maes)):.4f}")
        print("=======================================================")
        summary_path = os.path.join(args.output_dir, "ood_nicomplex_summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[Eval-NiComplex-OOD] Summary saved: {summary_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
