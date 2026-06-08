"""
NiComplex OOD multi-GPU batch inference.

Shards the OOD test set across GPUs; each GPU runs batched generation locally.

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -u OOD/NiComplex/inference_nicomplex_ood_multigpu.py \\
    --experiment train_rest_test_Pybox \\
    --Stage3_ckpt /data/jingyuan_data/NiComplex_OOD_Models/exp_train_rest_test_Pybox/checkpoint-1400 \\
    --gpus 0,1,2,3 \\
    --batch_size 4 \\
    --save_json /data/jingyuan_data/NiComplex_OOD_Models/exp_train_rest_test_Pybox/ood_test_predictions_checkpoint-1400.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
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
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel
from train_defaults import NICOMPLEX_DEFAULTS, VASKA_DEFAULTS
from utils import (
    OBJECT_REF_CHAT_SEP,
    UNIMOL_MAX_SEQ_LEN,
    _atoms_coords_remove_h_center,
    build_batch_multi,
    extract_single_token_repr,
    format_instruction_field,
)

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
    for idx, sample in enumerate(ood_samples):
        y = float(sample["ddG"])
        atoms = sample["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(sample["coordinates"], dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
        rows.append(
            {
                "idx": idx,
                "atoms": atoms,
                "coordinates": coords,
                "user_content": _build_user_content(sample),
                "ref": float(y),
                "R_Type": sample.get("R_Type"),
                "L_Scaffold": sample.get("L_Scaffold"),
            }
        )
    return rows


def _split_chat_prefix(tokenizer, user_content: str) -> Tuple[str, str]:
    prefix_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sep = OBJECT_REF_CHAT_SEP
    if sep not in prefix_str:
        return prefix_str, ""
    before_3d_str, rest = prefix_str.split(sep, 1)
    return before_3d_str, sep + rest


def _pad_token_ids(ids_list: Sequence[torch.Tensor], pad_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.shape[1] for x in ids_list)
    batch = []
    mask = []
    for ids in ids_list:
        cur = ids.to(device)
        pad_len = max_len - cur.shape[1]
        if pad_len > 0:
            cur = torch.nn.functional.pad(cur, (0, pad_len), value=pad_id)
        batch.append(cur)
        m = torch.ones(cur.shape, dtype=torch.long, device=device)
        if pad_len > 0:
            m[:, -pad_len:] = 0
        mask.append(m)
    return torch.cat(batch, dim=0), torch.cat(mask, dim=0)


@torch.inference_mode()
def generate_batch_with_single_token_structure(
    model: MultimodalModel,
    tokenizer,
    batch_atoms: List[list],
    batch_coords: List[np.ndarray],
    batch_user_contents: List[str],
    *,
    max_new_tokens: int = 64,
) -> List[str]:
    if not batch_atoms:
        return []

    device = next(model.llm.parameters()).device
    embed_layer = model.llm.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    batch_dict = build_batch_multi(
        batch_atoms,
        batch_coords,
        model.dictionary,
        max_seq_len=UNIMOL_MAX_SEQ_LEN,
        pad_idx=model._pad_idx,
        single_token_idx=model._single_token_idx,
        eos_idx=model._eos_idx,
        device=str(device),
    )
    model.unimol.eval()
    encoder_rep, _ = model.unimol(
        batch_dict["src_tokens"],
        batch_dict["src_distance"],
        batch_dict["src_coord"],
        batch_dict["src_edge_type"],
    )
    single_token_repr = extract_single_token_repr(encoder_rep)
    proj_dtype = next(model.single_token_projection_layer.parameters()).dtype
    single_token_repr = single_token_repr.to(dtype=proj_dtype)
    mol_embeds = model.single_token_projection_layer(single_token_repr).unsqueeze(1)

    before_parts, after_parts = zip(*[_split_chat_prefix(tokenizer, uc) for uc in batch_user_contents])
    before_ids_list = [tokenizer(b, return_tensors="pt").input_ids for b in before_parts]
    after_ids_list = [
        tokenizer(a, return_tensors="pt").input_ids if a else torch.zeros(1, 0, dtype=torch.long)
        for a in after_parts
    ]

    before_ids, before_mask = _pad_token_ids(before_ids_list, pad_id, device)
    after_ids, after_mask = _pad_token_ids(after_ids_list, pad_id, device)
    before_embeds = embed_layer(before_ids)
    after_embeds = embed_layer(after_ids)

    start_ids = torch.full((len(batch_atoms), 1), model._start_3d_id, dtype=torch.long, device=device)
    end_ids = torch.full((len(batch_atoms), 1), model._end_3d_id, dtype=torch.long, device=device)
    start_emb = embed_layer(start_ids)
    end_emb = embed_layer(end_ids)

    model_dtype = before_embeds.dtype
    start_emb = start_emb.to(model_dtype)
    end_emb = end_emb.to(model_dtype)
    mol_embeds = mol_embeds.to(model_dtype)
    three_d_block = torch.cat([start_emb, mol_embeds, end_emb], dim=1)
    three_d_mask = torch.ones((len(batch_atoms), three_d_block.shape[1]), dtype=torch.long, device=device)

    fused_embeddings = torch.cat([before_embeds, three_d_block, after_embeds], dim=1)
    fused_attention_mask = torch.cat([before_mask, three_d_mask, after_mask], dim=1)

    eos_id = tokenizer.eos_token_id
    prompt_lens = fused_attention_mask.sum(dim=1).tolist()
    model.llm.eval()

    out_ids = model.llm.generate(
        inputs_embeds=fused_embeddings,
        attention_mask=fused_attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )

    texts = []
    for row, prompt_len in zip(out_ids, prompt_lens):
        if row.shape[0] > prompt_len:
            gen_ids = row[prompt_len:]
        else:
            gen_ids = row
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return texts


def _worker(
    rank: int,
    world_size: int,
    gpu_ids: List[int],
    worker_args: dict,
    test_data: list,
    partial_path: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[rank])
    torch.set_grad_enabled(False)

    shard = test_data[rank::world_size]
    if not shard:
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    tokenizer = AutoTokenizer.from_pretrained(worker_args["model_name"])
    model = MultimodalModel(
        worker_args["model_name"],
        worker_args["three_d_encoder_dict"],
        recipe=RECIPE_STAGE3,
        init_ckpt=worker_args["stage3_ckpt"],
        train_3d_encoder=False,
        train_projection=False,
        train_lora=False,
        load_pretrained_projection=True,
        load_pretrained_lora=True,
        lora_r=worker_args["lora_r"],
        lora_alpha=worker_args["lora_alpha"],
        lora_target=worker_args["lora_target"],
    )
    model.eval()

    pred_records = []
    batch_size = worker_args["batch_size"]
    print(f"[Worker {rank}] GPU={gpu_ids[rank]} | shard={len(shard)} | batch_size={batch_size}", flush=True)

    for start in range(0, len(shard), batch_size):
        chunk = shard[start : start + batch_size]
        outputs = generate_batch_with_single_token_structure(
            model,
            tokenizer,
            [x["atoms"] for x in chunk],
            [x["coordinates"] for x in chunk],
            [x["user_content"] for x in chunk],
            max_new_tokens=worker_args["max_new_tokens"],
        )
        for sample, out in zip(chunk, outputs):
            pred = _parse_first_float(out)
            pred_records.append(
                {
                    "idx": sample["idx"],
                    "ref": sample["ref"],
                    "R_Type": sample["R_Type"],
                    "L_Scaffold": sample["L_Scaffold"],
                    "pred_text": out,
                    "pred_value": float(pred) if pred is not None and np.isfinite(pred) else None,
                }
            )
        done = min(start + batch_size, len(shard))
        if worker_args["print_every"] > 0 and done % worker_args["print_every"] == 0:
            parsed = sum(r["pred_value"] is not None for r in pred_records)
            print(f"[Worker {rank}] {done}/{len(shard)} | parsed={parsed}", flush=True)

    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(pred_records, f, ensure_ascii=False, indent=2)
    print(f"[Worker {rank}] saved partial predictions: {partial_path}", flush=True)


def _merge_and_metrics(pred_records: list, spec: dict, ckpt: str, experiment_name: str, n_ood_test: int) -> dict:
    pred_records = sorted(pred_records, key=lambda x: x["idx"])
    y_true, y_pred = [], []
    n_parse_fail = 0
    for rec in pred_records:
        if rec["pred_value"] is None:
            n_parse_fail += 1
        else:
            y_true.append(rec["ref"])
            y_pred.append(rec["pred_value"])

    if not y_pred:
        raise RuntimeError("No valid numeric predictions.")

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    mae = _mae(y_true_arr, y_pred_arr)
    r2 = _r2(y_true_arr, y_pred_arr)

    return {
        "experiment": experiment_name,
        "split_col": spec["split_col"],
        "train_types": sorted(spec["train_types"]),
        "test_types": sorted(spec["test_types"]),
        "ckpt": ckpt,
        "n_ood_test": n_ood_test,
        "n_parsed": len(y_pred_arr),
        "n_parse_fail": n_parse_fail,
        "mae": mae,
        "r2": r2 if np.isfinite(r2) else None,
        "predictions": [{k: v for k, v in rec.items() if k != "idx"} for rec in pred_records],
    }


def main():
    p = argparse.ArgumentParser(description="NiComplex OOD multi-GPU batch inference")
    p.add_argument("--model_name", type=str, default=VASKA_DEFAULTS["model_name"])
    p.add_argument("--3D_encoder_dict", dest="three_d_encoder_dict", type=str, default=VASKA_DEFAULTS["3D_encoder_dict"])
    p.add_argument("--Stage3_ckpt", dest="stage3_ckpt", type=str, required=True)
    p.add_argument("--lmdb", action="append", default=None, dest="lmdb_paths", metavar="PATH")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--experiment", type=str, required=True, choices=list(EXPERIMENT_NAMES))
    p.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    p.add_argument("--lora_r", type=int, default=NICOMPLEX_DEFAULTS["lora_r"])
    p.add_argument("--lora_alpha", type=int, default=NICOMPLEX_DEFAULTS["lora_alpha"])
    p.add_argument("--lora_target", type=str, default=NICOMPLEX_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"])
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--save_json", type=str, required=True)
    args = p.parse_args()

    if not os.path.isdir(args.stage3_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.stage3_ckpt}")

    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible.strip():
            gpu_ids = [int(x.strip()) for x in visible.split(",") if x.strip()]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise RuntimeError("No GPUs specified. Set --gpus or CUDA_VISIBLE_DEVICES.")

    lmdb_paths = args.lmdb_paths or [DEFAULT_LMDB]
    print(f"[Eval-NiComplex-OOD-MG] Reading LMDB: {lmdb_paths}")
    all_valid, _ = load_merged_valid_nicomplex_records(lmdb_paths, local_rank=0)
    _, ood_samples, spec = split_by_experiment(all_valid, args.experiment)
    test_data = _collect_eval_samples(ood_samples)
    print(
        f"[Eval-NiComplex-OOD-MG] experiment={args.experiment} | ood_test={len(test_data)} | "
        f"gpus={gpu_ids} | batch_size={args.batch_size} | ckpt={args.stage3_ckpt}"
    )

    worker_args = {
        "model_name": args.model_name,
        "three_d_encoder_dict": args.three_d_encoder_dict,
        "stage3_ckpt": args.stage3_ckpt,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target": args.lora_target,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "print_every": args.print_every,
    }

    save_json = os.path.abspath(args.save_json)
    partial_dir = os.path.join(os.path.dirname(save_json) or ".", "_partial_multigpu")
    os.makedirs(partial_dir, exist_ok=True)
    partial_paths = [os.path.join(partial_dir, f"rank{r}.json") for r in range(len(gpu_ids))]

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    world_size = len(gpu_ids)
    for rank in range(world_size):
        proc = mp.Process(
            target=_worker,
            args=(rank, world_size, gpu_ids, worker_args, test_data, partial_paths[rank]),
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Worker failed with exit code {proc.exitcode}")

    merged_records = []
    for path in partial_paths:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            merged_records.extend(json.load(f))

    result = _merge_and_metrics(
        merged_records,
        spec,
        args.stage3_ckpt,
        args.experiment,
        len(test_data),
    )
    os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n--- OOD experiment={args.experiment} ---")
    print(
        f"MAE: {result['mae']:.6f} kcal/mol | R2: {result['r2']:.6f} | "
        f"N={result['n_parsed']} | parse_fail={result['n_parse_fail']}"
    )
    print(f"[Eval-NiComplex-OOD-MG] Saved {save_json}")


if __name__ == "__main__":
    main()
