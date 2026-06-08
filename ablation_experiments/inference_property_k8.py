"""
Evaluate Property-K8 checkpoint on tmQMg test LMDB (MAE / R²).

Example:
  CUDA_VISIBLE_DEVICES=0 python ablation_experiments/inference_property_k8.py \\
    --Stage3_ckpt /data/jingyuan_data/Stage3_Property_dipole_moment_k8_ckpt/checkpoint-3000 \\
    --test_lmdb /data/jingyuan_data/tmqmg/stage3/test/tmqmg_atom_only_new.lmdb \\
    --property dipole_moment
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
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_STAGE3_DIR = os.path.join(_PROJECT_ROOT, "Stage3")
if _STAGE3_DIR not in sys.path:
    sys.path.insert(0, _STAGE3_DIR)

import utils  # noqa: F401
from Property import PROPERTY_CONFIG, TmQMgSingleTokenUnimolDataset
from multimodal_LLM import RECIPE_STAGE3
from train_defaults import VASKA_DEFAULTS
from utils import OBJECT_REF_CHAT_SEP, format_instruction_field

from ablation_experiments.multimodal_k8 import MultimodalModelK8

DEFAULT_TEST_LMDB = "/data/jingyuan_data/tmqmg/stage3/test/tmqmg_atom_only_new.lmdb"
DEFAULT_CKPT = "/data/jingyuan_data/Stage3_Property_dipole_moment_k8_ckpt/checkpoint-3000"


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


def _generate_with_k8_structure_greedy(
    model: MultimodalModelK8,
    tokenizer,
    atoms,
    coords,
    user_content: str,
    *,
    max_new_tokens: int = 64,
    repetition_penalty: float = 1.05,
) -> str:
    device = next(model.llm.parameters()).device
    embed_layer = model.llm.get_input_embeddings()

    model.unimol.eval()
    with torch.inference_mode():
        atom_repr, atom_pad_mask = model._encode_atom_repr([atoms], [coords], device)
        proj_dtype = next(model.k8_projection_layer.parameters()).dtype
        atom_repr = atom_repr.to(dtype=proj_dtype)
        structure_proj = model.k8_projection_layer(atom_repr, atom_pad_mask.to(device)).to(
            dtype=embed_layer.weight.dtype
        )

    prefix_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sep = OBJECT_REF_CHAT_SEP
    before_3d_str = prefix_str.split(sep, 1)[0]
    after_3d_str = sep + prefix_str.split(sep, 1)[1]

    with torch.inference_mode():
        before_3d_ids = tokenizer(before_3d_str, return_tensors="pt").input_ids.to(device)
        after_3d_ids = tokenizer(after_3d_str, return_tensors="pt").input_ids.to(device)
        before_3d_embeds = embed_layer(before_3d_ids)
        after_3d_embeds = embed_layer(after_3d_ids)
        start_emb = embed_layer(torch.tensor([[model._start_3d_id]], device=device))
        end_emb = embed_layer(torch.tensor([[model._end_3d_id]], device=device))

    model_dtype = before_3d_embeds.dtype
    start_emb = start_emb.to(model_dtype)
    end_emb = end_emb.to(model_dtype)
    structure_proj = structure_proj.to(model_dtype)
    three_d_block = torch.cat([start_emb, structure_proj, end_emb], dim=1)

    fused_embeddings = torch.cat([before_3d_embeds, three_d_block, after_3d_embeds], dim=1)
    fused_attention_mask = torch.ones((1, fused_embeddings.shape[1]), dtype=torch.long, device=device)

    seq_len = fused_embeddings.shape[1]
    eos_id = tokenizer.eos_token_id
    generated_ids = []
    current_embeds = fused_embeddings
    current_attention = fused_attention_mask
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    model.llm.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model.llm(
                inputs_embeds=current_embeds,
                attention_mask=current_attention,
                position_ids=position_ids,
                return_dict=True,
            )
        logits = outputs.logits[:, -1, :].float().clone()
        for gid in generated_ids:
            if logits[0, gid] > 0:
                logits[0, gid] /= repetition_penalty
            else:
                logits[0, gid] *= repetition_penalty
        next_id = int(logits.argmax(dim=-1).item())
        generated_ids.append(next_id)
        if eos_id is not None and next_id == eos_id and len(generated_ids) >= 1:
            break

        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        next_embed = embed_layer(next_token).to(current_embeds.dtype)
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        current_attention = torch.cat(
            [current_attention, torch.ones((1, 1), dtype=torch.long, device=device)],
            dim=1,
        )
        position_ids = torch.cat(
            [position_ids, torch.tensor([[position_ids.shape[1]]], device=device, dtype=torch.long)],
            dim=1,
        )

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _collect_eval_samples(dataset: TmQMgSingleTokenUnimolDataset):
    import pickle

    samples = []
    prop_key = dataset.property_key
    for lmdb_path, key_bytes in dataset.key_index:
        env = dataset._envs.get(lmdb_path)
        if env is None:
            import lmdb

            from utils import _lmdb_env_kwargs

            env = lmdb.open(lmdb_path, **_lmdb_env_kwargs())
            dataset._envs[lmdb_path] = env
        with env.begin() as txn:
            data = pickle.loads(txn.get(key_bytes))
        smiles = format_instruction_field(data.get("smiles"))
        y = float(data[prop_key])
        atoms = data["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(data["coordinates"], dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        from utils import _atoms_coords_remove_h_center

        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
        user_content = f"{dataset.instruction} {smiles}"
        samples.append((atoms, coords, user_content, y))
    return samples


def main():
    p = argparse.ArgumentParser(description="Property-K8 test LMDB evaluation (MAE/R²).")
    p.add_argument("--model_name", type=str, default=VASKA_DEFAULTS["model_name"])
    p.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=VASKA_DEFAULTS["3D_encoder_dict"],
    )
    p.add_argument("--Stage3_ckpt", dest="stage3_ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--test_lmdb", type=str, default=DEFAULT_TEST_LMDB)
    p.add_argument("--property", type=str, default="dipole_moment", choices=list(PROPERTY_CONFIG.keys()))
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_target", type=str, default="all", choices=["qv", "qkv", "all"])
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--print_every", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=None, help="Limit test samples (debug)")
    p.add_argument("--save_json", type=str, default=None)
    args = p.parse_args()

    if not os.path.isfile(args.test_lmdb):
        raise FileNotFoundError(f"Test LMDB not found: {args.test_lmdb}")
    if not os.path.isdir(args.stage3_ckpt):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.stage3_ckpt}")

    prop_cfg = PROPERTY_CONFIG[args.property]
    instruction_use = prop_cfg.get("instruction_smiles") or prop_cfg.get("instruction_description")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalModelK8(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        init_ckpt=args.stage3_ckpt,
        train_3d_encoder=False,
        train_projection=False,
        train_lora=False,
        load_pretrained_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
    )
    model.eval()

    dataset = TmQMgSingleTokenUnimolDataset(
        [args.test_lmdb],
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        property_key=prop_cfg["key"],
        instruction=instruction_use,
        use_polished_description=False,
    )
    test_data = _collect_eval_samples(dataset)
    print(f"[Eval-K8] test_lmdb={args.test_lmdb} | valid={len(test_data)} | ckpt={args.stage3_ckpt}")

    y_true, y_pred = [], []
    n_parse_fail = 0
    pred_records = []

    for i, (atoms, coords, user_content, y) in enumerate(test_data, start=1):
        out = _generate_with_k8_structure_greedy(
            model=model,
            tokenizer=tokenizer,
            atoms=atoms,
            coords=coords,
            user_content=user_content,
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
            print(f"[Eval-K8] {i}/{len(test_data)} | parsed={len(y_pred)} | parse_fail={n_parse_fail}")

    if len(y_pred) == 0:
        raise RuntimeError("No valid numeric predictions parsed from model outputs.")

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    mae = _mae(y_true_arr, y_pred_arr)
    r2 = _r2(y_true_arr, y_pred_arr)
    unit = prop_cfg.get("unit", "")

    print("\n================ Property-K8 Test Metrics ================")
    print(f"property:       {args.property} ({unit})")
    print(f"test_total:     {len(test_data)}")
    print(f"pred_parsed:    {len(y_pred_arr)}")
    print(f"parse_fail:     {n_parse_fail}")
    print(f"MAE:            {mae:.6f}")
    print(f"R2:             {r2:.6f}" if np.isfinite(r2) else "R2:             nan (zero variance in y_true)")
    print("==========================================================")

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "property": args.property,
                    "ckpt": args.stage3_ckpt,
                    "test_lmdb": args.test_lmdb,
                    "mae": mae,
                    "r2": r2 if np.isfinite(r2) else None,
                    "n_parsed": len(y_pred_arr),
                    "n_parse_fail": n_parse_fail,
                    "predictions": pred_records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[Eval-K8] Saved: {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
