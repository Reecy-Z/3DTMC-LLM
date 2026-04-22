"""
Evaluate Stage3 Vaska model on the test split (same split rule as Stage3/Vaska_Complex.py).

Metrics reported:
  - MAE
  - R2

Example:
  CUDA_VISIBLE_DEVICES=0 python inference_Vaska_Complex.py \
    --model_name /path/to/HF_models/Qwen3-4B-Instruct-2507 \
    --3D_encoder_dict /path/to/3D_encoder_dict.txt \
    --Stage3_ckpt /path/to/Vaska/checkpoint \
    --lmdb /path/to/vaskas-space/data.lmdb \
    --split_seed 43
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

import utils  # noqa: F401
from Stage3.Vaska_Complex import VASKA_INSTRUCTION, read_vaska_lmdb
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel
from train_defaults import VASKA_DEFAULTS
from utils import (
    OBJECT_REF_CHAT_SEP,
    UNIMOL_MAX_SEQ_LEN,
    _atoms_coords_remove_h_center,
    build_batch_multi,
    extract_single_token_repr,
    format_instruction_field,
)


def _build_user_content(smiles: str) -> str:
    smiles = format_instruction_field(smiles)
    return f"{VASKA_INSTRUCTION} {smiles}".strip()


def _generate_with_single_token_structure_greedy(
    model: MultimodalModel,
    tokenizer,
    atoms,
    coords,
    user_content: str,
    *,
    max_new_tokens: int = 64,
    repetition_penalty: float = 1.1,
) -> str:
    device = next(model.llm.parameters()).device
    embed_layer = model.llm.get_input_embeddings()

    batch_dict = build_batch_multi(
        [atoms],
        [coords],
        model.dictionary,
        max_seq_len=UNIMOL_MAX_SEQ_LEN,
        pad_idx=model._pad_idx,
        single_token_idx=model._single_token_idx,
        eos_idx=model._eos_idx,
        device=str(device),
    )
    model.unimol.eval()
    with torch.inference_mode():
        encoder_rep, _ = model.unimol(
            batch_dict["src_tokens"],
            batch_dict["src_distance"],
            batch_dict["src_coord"],
            batch_dict["src_edge_type"],
        )
    single_token_repr = extract_single_token_repr(encoder_rep)
    proj_dtype = next(model.single_token_projection_layer.parameters()).dtype
    single_token_repr = single_token_repr.to(dtype=proj_dtype)
    with torch.inference_mode():
        mol_embeds = model.single_token_projection_layer(single_token_repr).unsqueeze(1)

    prefix_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sep = OBJECT_REF_CHAT_SEP
    # Keep the same split style as the legacy generate() implementation.
    before_3d_str = prefix_str.split(sep, 1)[0]
    after_3d_str = sep + prefix_str.split(sep, 1)[1]

    with torch.inference_mode():
        before_3d_ids = tokenizer(before_3d_str, return_tensors="pt").input_ids.to(device)
        after_3d_ids = tokenizer(after_3d_str, return_tensors="pt").input_ids.to(device)
        before_3d_embeds = embed_layer(before_3d_ids)
        after_3d_embeds = embed_layer(after_3d_ids)
        start_ids = tokenizer("<|object_ref_start|>", return_tensors="pt").input_ids.to(device)
        end_ids = tokenizer("<|object_ref_end|>", return_tensors="pt").input_ids.to(device)
        start_emb = embed_layer(start_ids)
        end_emb = embed_layer(end_ids)

    model_dtype = before_3d_embeds.dtype
    start_emb = start_emb.to(model_dtype)
    end_emb = end_emb.to(model_dtype)
    mol_embeds = mol_embeds.to(model_dtype)
    three_d_block = torch.cat([start_emb, mol_embeds, end_emb], dim=1)

    fused_embeddings = torch.cat([before_3d_embeds, three_d_block, after_3d_embeds], dim=1)
    mask_before = torch.ones((1, before_3d_embeds.shape[1]), dtype=torch.long, device=device)
    mask_3d = torch.ones((1, three_d_block.shape[1]), dtype=torch.long, device=device)
    mask_after = torch.ones((1, after_3d_embeds.shape[1]), dtype=torch.long, device=device)
    fused_attention_mask = torch.cat([mask_before, mask_3d, mask_after], dim=1)

    seq_len = fused_embeddings.shape[1]
    eos_id = tokenizer.eos_token_id
    min_new_tokens = 1
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
        if eos_id is not None and next_id == eos_id and len(generated_ids) >= min_new_tokens:
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


def _parse_first_float(text: str) -> Optional[float]:
    if not text:
        return None
    # Supports integers, decimals, and scientific notation.
    m = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _build_test_split(raw_samples, split_seed: int):
    n_total = len(raw_samples)
    idx = np.arange(n_total)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(idx)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    test_idx = idx[n_train + n_val :]
    return [raw_samples[i] for i in test_idx]


def _filter_valid_test_samples(samples):
    valid = []
    for d in samples:
        if "atoms" not in d or "coordinates" not in d or "barrier" not in d:
            continue
        try:
            y = float(d["barrier"])
            if np.isnan(y):
                continue
        except Exception:
            continue
        smiles = format_instruction_field(d.get("smiles"))
        if not smiles:
            continue
        atoms = d["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(d["coordinates"], dtype=np.float32)
        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
        if len(atoms) == 0:
            continue
        valid.append((atoms, coords, smiles, y))
    return valid


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main():
    p = argparse.ArgumentParser(
        description="Inference-only evaluation for Vaska test split (MAE/R2).",
    )
    p.add_argument("--model_name", type=str, default=VASKA_DEFAULTS["model_name"])
    p.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=VASKA_DEFAULTS["3D_encoder_dict"],
    )
    p.add_argument("--Stage3_ckpt", dest="stage3_ckpt", type=str)
    p.add_argument("--lmdb", type=str, default=VASKA_DEFAULTS["lmdb"])
    p.add_argument("--split_seed", type=int, default=VASKA_DEFAULTS["split_seed"])
    p.add_argument("--lora_r", type=int, default=VASKA_DEFAULTS["lora_r"])
    p.add_argument("--lora_alpha", type=int, default=VASKA_DEFAULTS["lora_alpha"])
    p.add_argument("--lora_target", type=str, default=VASKA_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"])
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save per-sample test predictions as JSON list.",
    )
    args = p.parse_args()

    if not os.path.isfile(args.lmdb):
        raise FileNotFoundError(f"LMDB not found: {args.lmdb}")
    if not args.stage3_ckpt or not os.path.isdir(args.stage3_ckpt):
        raise FileNotFoundError(f"Stage3 checkpoint directory not found: {args.stage3_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalModel(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        init_ckpt=args.stage3_ckpt,
        train_3d_encoder=False,
        train_projection=False,
        train_lora=False,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
    )
    model.eval()

    print(f"[Eval] Reading LMDB: {args.lmdb}")
    raw = read_vaska_lmdb(args.lmdb, max_samples=None, show_progress=True)
    if len(raw) == 0:
        raise RuntimeError("No LMDB records loaded.")

    test_raw = _build_test_split(raw, args.split_seed)
    test_data = _filter_valid_test_samples(test_raw)
    if len(test_data) == 0:
        raise RuntimeError("No valid test samples after filtering.")

    print(f"[Eval] total={len(raw)} | test_raw={len(test_raw)} | test_valid={len(test_data)}")

    y_true = []
    y_pred = []
    n_parse_fail = 0
    pred_records = []

    for i, (atoms, coords, smiles, y) in enumerate(test_data, start=1):
        user_content = _build_user_content(smiles)
        out = _generate_with_single_token_structure_greedy(
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
            print(f"[Eval] processed {i}/{len(test_data)} | parsed={len(y_pred)} | parse_fail={n_parse_fail}")

    if len(y_pred) == 0:
        raise RuntimeError("No valid numeric predictions parsed from model outputs.")

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    mae = _mae(y_true_arr, y_pred_arr)
    r2 = _r2(y_true_arr, y_pred_arr)

    print("\n================ Vaska Test Metrics ================")
    print(f"split_seed:     {args.split_seed}")
    print(f"test_total:     {len(test_data)}")
    print(f"pred_parsed:    {len(y_pred_arr)}")
    print(f"parse_fail:     {n_parse_fail}")
    print(f"MAE:            {mae:.6f}")
    print(f"R2:             {r2:.6f}" if np.isfinite(r2) else "R2:             nan (zero variance in y_true)")
    print("====================================================")

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pred_records, f, ensure_ascii=False, indent=2)
        print(f"[Eval] Saved predictions JSON: {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

