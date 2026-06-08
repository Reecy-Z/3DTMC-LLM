"""
Evaluate Vaska ligand-OOD checkpoint on held-out ligand test set (MAE / R²).

OOD test = complexes where the held-out ligand appears in any of ligand_a1/a2/b/c.

Example:
  CUDA_VISIBLE_DEVICES=0 python -u OOD/Vaska/inference_vaska_ligand_ood.py \\
    --holdout_ligand dft-co \\
    --Stage3_ckpt /data/jingyuan_data/Vaska_Ligand_OOD_Models/ligand_dft-co/checkpoint-xxx
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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
from Vaska_Complex import read_vaska_lmdb
from inference_Vaska_Complex import (
    _build_user_content,
    _filter_valid_test_samples,
    _generate_with_single_token_structure_greedy,
    _mae,
    _parse_first_float,
    _r2,
)
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel
from train_defaults import VASKA_DEFAULTS

from OOD.Vaska.ligand_split import LIGANDS, ligand_dirname, lolo_split_by_ligand, summarize_ligand_presence

DEFAULT_OUTPUT_DIR = "/data/jingyuan_data/Vaska_Ligand_OOD_Models"


def _eval_one_fold(args, holdout_ligand: str, all_raw: list) -> dict:
    ckpt = args.stage3_ckpt
    if not ckpt:
        ckpt = os.path.join(args.output_dir, ligand_dirname(holdout_ligand))
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(f"Checkpoint not found for {holdout_ligand}: {ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalModel(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        init_ckpt=ckpt,
        train_3d_encoder=False,
        train_projection=False,
        train_lora=False,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
    )
    model.eval()

    _, ood_raw = lolo_split_by_ligand(all_raw, holdout_ligand)
    test_data = _filter_valid_test_samples(ood_raw)
    if not test_data:
        raise RuntimeError(f"No valid OOD test samples for ligand={holdout_ligand}")

    print(
        f"[Eval-Vaska-Ligand-OOD] holdout={holdout_ligand} | ood_raw={len(ood_raw)} | "
        f"valid={len(test_data)} | ckpt={ckpt}"
    )

    y_true, y_pred = [], []
    n_parse_fail = 0
    pred_records = []

    for i, (atoms, coords, smiles, y) in enumerate(test_data, start=1):
        out = _generate_with_single_token_structure_greedy(
            model=model,
            tokenizer=tokenizer,
            atoms=atoms,
            coords=coords,
            user_content=_build_user_content(smiles),
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
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
            print(f"[Eval-Vaska-Ligand-OOD] {i}/{len(test_data)} | parsed={len(y_pred)} | parse_fail={n_parse_fail}")

    if not y_pred:
        raise RuntimeError("No valid numeric predictions.")

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    mae = _mae(y_true_arr, y_pred_arr)
    r2 = _r2(y_true_arr, y_pred_arr)

    print(f"\n--- OOD ligand={holdout_ligand} ---")
    print(f"MAE: {mae:.6f} kcal/mol | R2: {r2:.6f} | N={len(y_pred_arr)} | parse_fail={n_parse_fail}")

    result = {
        "holdout_ligand": holdout_ligand,
        "split": "leave_one_ligand_out",
        "ckpt": ckpt,
        "n_ood_test": len(test_data),
        "n_parsed": len(y_pred_arr),
        "n_parse_fail": n_parse_fail,
        "mae": mae,
        "r2": r2 if np.isfinite(r2) else None,
        "predictions": pred_records,
    }

    if args.save_json or args.run_all_loops:
        if args.run_all_loops:
            out_path = os.path.join(args.output_dir, ligand_dirname(holdout_ligand), "ood_test_predictions.json")
        else:
            out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Eval-Vaska-Ligand-OOD] Saved {out_path}")

    return result


def main():
    p = argparse.ArgumentParser(description="Vaska ligand-OOD inference (LOLO test set)")
    p.add_argument("--model_name", type=str, default=VASKA_DEFAULTS["model_name"])
    p.add_argument("--3D_encoder_dict", dest="three_d_encoder_dict", type=str, default=VASKA_DEFAULTS["3D_encoder_dict"])
    p.add_argument("--Stage3_ckpt", dest="stage3_ckpt", type=str, default=None)
    p.add_argument("--lmdb", type=str, default=VASKA_DEFAULTS["lmdb"])
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--holdout_ligand", type=str, default=None, choices=list(LIGANDS))
    p.add_argument("--run_all_loops", action="store_true")
    p.add_argument("--lora_r", type=int, default=VASKA_DEFAULTS["lora_r"])
    p.add_argument("--lora_alpha", type=int, default=VASKA_DEFAULTS["lora_alpha"])
    p.add_argument("--lora_target", type=str, default=VASKA_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"])
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--save_json", type=str, default=None)
    args = p.parse_args()

    if not args.run_all_loops and not args.holdout_ligand:
        p.error("Specify --holdout_ligand or --run_all_loops")

    print(f"[Eval-Vaska-Ligand-OOD] Reading LMDB: {args.lmdb}")
    all_raw = read_vaska_lmdb(args.lmdb, max_samples=None, show_progress=True)
    print(f"[Eval-Vaska-Ligand-OOD] ligand presence counts: {summarize_ligand_presence(all_raw)}")

    folds = list(LIGANDS) if args.run_all_loops else [args.holdout_ligand]
    summary = []
    for holdout in folds:
        summary.append(_eval_one_fold(args, holdout, all_raw))

    if len(summary) > 1:
        print("\n================ OOD ligand summary ================")
        for row in summary:
            r2s = "nan" if row["r2"] is None else f"{row['r2']:.4f}"
            print(f"  {row['holdout_ligand']:14s}  MAE={row['mae']:.4f}  R2={r2s}  N={row['n_parsed']}")
        maes = [r["mae"] for r in summary]
        print(f"  mean MAE across folds: {float(np.mean(maes)):.4f}")
        print("====================================================")
        summary_path = os.path.join(args.output_dir, "ood_ligand_summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[Eval-Vaska-Ligand-OOD] Summary saved: {summary_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
