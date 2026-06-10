"""
Unified inference: description ablations (multi-GPU batch) and property evaluation (MAE / R²).

Library usage:
  from inference import collect_description_samples, generate_batch, eval_property_lmdb

CLI:
  python inference.py description --mode stage2 --ckpt ... --save_json ...
  python inference.py description --mode multi_token --ckpt ... --save_json ...
  python inference.py stage3 --task dipole_moment --Stage3_ckpt ... --split_csv split.csv  # OOD
  python inference.py stage3 --task dipole_moment --fixed_lmdb_eval --Stage3_ckpt ... --test_lmdb ...
  python inference.py stage3 --task vaska_barrier --holdout_ligand dft-co --lmdb ... --Stage3_ckpt ...
  python inference.py stage3 --task nicomplex_ddg --ood_experiment train_rest_test_Pybox --Stage3_ckpt ...
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import utils  # noqa: F401
from task_datasets import (
    INSTRUCTION_3D_ONLY,
    INSTRUCTION_DESCRIPTION,
    INSTRUCTION_DESCRIPTION_3D_ONLY,
    NI_INSTRUCTION,
    PROPERTY_CONFIG,
    RESPONSE_KEY,
    TmQMg3DOnlyUnimolDataset,
    TmQMgSingleTokenUnimolDataset,
    atoms_coords_from_record,
    filter_valid_vaska_records,
    load_merged_valid_nicomplex_records,
    load_merged_valid_vaska_records,
    read_vaska_lmdb,
)
from task_registry import STAGE3_TASKS, get_task, normalize_task_name, resolve_instruction, resolve_user_content
from multimodal_LLM import (
    RECIPE_STAGE3,
    MultimodalModel,
    MultimodalModelFreeze3D,
    MultimodalModelMultiToken,
    MultimodalModelRandom3D,
    generate_batch_with_random_structure,
    generate_with_single_token_structure,
    generate_with_random_structure,
)
from train_defaults import DESCRIPTION_DEFAULTS, NICOMPLEX_DEFAULTS, VASKA_DEFAULTS
from utils import (
    OBJECT_REF_CHAT_SEP,
    UNIMOL_MAX_SEQ_LEN,
    _atoms_coords_remove_h_center,
    _lmdb_env_kwargs,
    build_batch_multi,
    extract_single_token_repr,
    format_instruction_field,
)

DESCRIPTION_MODES = ("stage2", "freeze_3d", "random_3d", "3d_only", "multi_token")
STAGE2_RESPONSE_KEY = "enriched_description"
STAGE3_INFER_MODES = ("single_token", "multi_token", "3d_only", "freeze_3d", "random_3d")
PROPERTY_MODES = STAGE3_INFER_MODES
TMQM_TASKS = frozenset({"dipole_moment", "polarisability", "homo_lumo_gap"})

DEFAULT_PROPERTY_OOD_LMDB = [
    "/path/to/tmqmg/stage3/train/tmqmg_atom_only_new.lmdb",
    "/path/to/tmqmg/stage3/test/tmqmg_atom_only_new.lmdb",
]
DEFAULT_NICOMPLEX_LMDB = "/path/to/NiComplex/data.lmdb"
DEFAULT_VASKA_LMDB = VASKA_DEFAULTS["lmdb"]

DESCRIPTION_MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "stage2": {
        "include_smiles": True,
        "instruction": INSTRUCTION_DESCRIPTION,
        "load_pretrained_projection": True,
        "response_key": STAGE2_RESPONSE_KEY,
    },
    "freeze_3d": {
        "include_smiles": True,
        "instruction": INSTRUCTION_DESCRIPTION,
        "load_pretrained_projection": True,
    },
    "random_3d": {
        "include_smiles": True,
        "instruction": INSTRUCTION_DESCRIPTION,
        "load_pretrained_projection": False,
    },
    "3d_only": {
        "include_smiles": False,
        "instruction": INSTRUCTION_DESCRIPTION_3D_ONLY,
        "load_pretrained_projection": True,
    },
    "multi_token": {
        "include_smiles": True,
        "instruction": INSTRUCTION_DESCRIPTION,
        "load_pretrained_projection": False,
    },
}

DEFAULT_PROPERTY_TEST_LMDB = "/path/to/tmqmg/stage3/test/tmqmg_atom_only_new.lmdb"


# ---------------------------------------------------------------------------
# Token / chat helpers
# ---------------------------------------------------------------------------


def split_chat_prefix(tokenizer, user_content: str) -> Tuple[str, str]:
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


def pad_token_ids(
    ids_list: Sequence[torch.Tensor], pad_id: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.shape[1] for x in ids_list)
    batch, mask = [], []
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


# ---------------------------------------------------------------------------
# Property metrics
# ---------------------------------------------------------------------------


def parse_first_float(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# LMDB sample collection
# ---------------------------------------------------------------------------


def build_user_content(instruction: str, smiles: Optional[str], include_smiles: bool) -> str:
    if include_smiles and smiles:
        return f"{instruction.strip()} {format_instruction_field(smiles)}".strip()
    return instruction.strip()


def collect_description_samples(
    lmdb_paths: List[str],
    *,
    include_smiles: bool,
    instruction: str,
    response_key: str = RESPONSE_KEY,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for path in lmdb_paths:
        if not path or not os.path.exists(path):
            continue
        env = lmdb.open(path, **_lmdb_env_kwargs())
        try:
            with env.begin() as txn:
                for _, value in txn.cursor():
                    data = pickle.loads(value)
                    if "atoms" not in data or "coordinates" not in data:
                        continue
                    if include_smiles and "smiles" not in data:
                        continue
                    ref = data.get(response_key)
                    if not ref or (isinstance(ref, str) and not ref.strip()):
                        continue
                    sample_idx = len(samples)
                    atoms = data["atoms"]
                    if isinstance(atoms, np.ndarray):
                        atoms = atoms.tolist()
                    atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
                    coords = np.asarray(data["coordinates"], dtype=np.float32)
                    if coords.ndim == 3:
                        coords = coords[0]
                    atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
                    smiles = data.get("smiles", "") if include_smiles else None
                    samples.append(
                        {
                            "sample_idx": sample_idx,
                            "lmdb_path": path,
                            "smiles": smiles,
                            "atoms": atoms,
                            "coordinates": coords,
                            "user_content": build_user_content(instruction, smiles, include_smiles),
                            "reference": str(ref).strip(),
                        }
                    )
                    if max_samples is not None and len(samples) >= max_samples:
                        return samples
        finally:
            env.close()
    if not samples:
        raise RuntimeError(f"No valid LMDB samples with {response_key}: {lmdb_paths}")
    return samples


def default_description_test_lmdb(test_lmdb: Optional[List[str]] = None) -> List[str]:
    if test_lmdb:
        return list(test_lmdb)
    default = DESCRIPTION_DEFAULTS.get("test_lmdb") or DESCRIPTION_DEFAULTS.get("val_lmdb")
    return [default] if isinstance(default, str) else list(default)


def collect_property_eval_samples(dataset) -> List[Tuple[list, np.ndarray, str, float]]:
    """Return (atoms, coords, user_content, y_true) from a property dataset."""
    samples = []
    prop_key = getattr(dataset, "property_key", None)
    task_name = getattr(dataset, "task_name", None) or prop_key
    use_polished = getattr(dataset, "use_polished_description", False)
    instruction_override = getattr(dataset, "instruction", None)
    mode = "3d_only" if isinstance(dataset, TmQMg3DOnlyUnimolDataset) else "single_token"
    for lmdb_path, key_bytes in dataset.key_index:
        env = dataset._envs.get(lmdb_path)
        if env is None:
            env = lmdb.open(lmdb_path, **_lmdb_env_kwargs())
            dataset._envs[lmdb_path] = env
        with env.begin() as txn:
            data = pickle.loads(txn.get(key_bytes))
        y = float(data[prop_key])
        atoms, coords = atoms_coords_from_record(data)
        user_content = resolve_user_content(
            task_name,
            data,
            mode=mode,
            use_polished_description=use_polished,
            instruction_override=instruction_override,
        )
        samples.append((atoms, coords, user_content, y))
    return samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_description_model(
    mode: str,
    model_name: str,
    three_d_encoder_dict: str,
    ckpt_dir: str,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_target: str,
    random_3d_seed: int,
):
    if mode not in DESCRIPTION_MODE_CONFIG:
        raise ValueError(f"Unknown mode={mode!r}; choose from {DESCRIPTION_MODES}")
    cfg = DESCRIPTION_MODE_CONFIG[mode]
    common = dict(
        model_name=model_name,
        three_d_encoder_dict_path=three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        init_ckpt=ckpt_dir,
        train_3d_encoder=False,
        train_projection=False,
        train_lora=False,
        load_pretrained_projection=cfg["load_pretrained_projection"],
        load_pretrained_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target=lora_target,
    )
    if mode in ("stage2", "3d_only"):
        return MultimodalModel(**common)
    if mode == "freeze_3d":
        return MultimodalModelFreeze3D(**common)
    if mode == "random_3d":
        return MultimodalModelRandom3D(**common, random_3d_seed=random_3d_seed)
    if mode == "multi_token":
        return MultimodalModelMultiToken(**common)
    raise ValueError(f"Unknown mode: {mode}")


def load_stage3_model(
    mode: str,
    model_name: str,
    three_d_encoder_dict: str,
    ckpt_dir: str,
    *,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_target: str = "all",
    random_3d_seed: int = DESCRIPTION_DEFAULTS["random_3d_seed"],
):
    if mode not in STAGE3_INFER_MODES:
        raise ValueError(f"Unknown mode={mode!r}; choose from {STAGE3_INFER_MODES}")
    common = dict(
        recipe=RECIPE_STAGE3,
        init_ckpt=ckpt_dir,
        train_3d_encoder=False,
        train_projection=False,
        train_lora=False,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target=lora_target,
    )
    if mode == "single_token":
        return MultimodalModel(
            model_name,
            three_d_encoder_dict,
            load_pretrained_projection=True,
            load_pretrained_lora=True,
            **common,
        )
    if mode == "multi_token":
        return MultimodalModelMultiToken(
            model_name,
            three_d_encoder_dict,
            load_pretrained_lora=True,
            **common,
        )
    if mode == "3d_only":
        return MultimodalModel(
            model_name,
            three_d_encoder_dict,
            load_pretrained_projection=True,
            load_pretrained_lora=True,
            **common,
        )
    if mode == "freeze_3d":
        return MultimodalModelFreeze3D(
            model_name,
            three_d_encoder_dict,
            load_pretrained_projection=True,
            load_pretrained_lora=True,
            **common,
        )
    if mode == "random_3d":
        return MultimodalModelRandom3D(
            model_name,
            three_d_encoder_dict,
            random_3d_seed=random_3d_seed,
            load_pretrained_projection=False,
            load_pretrained_lora=True,
            **common,
        )
    raise ValueError(f"Unhandled mode={mode!r}")


def load_property_model(
    mode: str,
    model_name: str,
    three_d_encoder_dict: str,
    ckpt_dir: str,
    *,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_target: str = "all",
):
    return load_stage3_model(
        mode,
        model_name,
        three_d_encoder_dict,
        ckpt_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target=lora_target,
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate_with_multi_token_structure_greedy(
    model: MultimodalModelMultiToken,
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
    atom_repr, atom_pad_mask = model._encode_atom_repr([atoms], [coords], device)
    proj_dtype = next(model.multi_token_projection_layer.parameters()).dtype
    atom_repr = atom_repr.to(dtype=proj_dtype)
    structure_proj = model.multi_token_projection_layer(atom_repr, atom_pad_mask.to(device)).to(
        dtype=embed_layer.weight.dtype
    )

    before_3d_str, after_3d_str = split_chat_prefix(tokenizer, user_content)
    before_3d_ids = tokenizer(before_3d_str, return_tensors="pt").input_ids.to(device)
    after_3d_ids = (
        tokenizer(after_3d_str, return_tensors="pt").input_ids.to(device)
        if after_3d_str
        else torch.zeros(1, 0, dtype=torch.long, device=device)
    )
    before_3d_embeds = embed_layer(before_3d_ids)
    after_3d_embeds = embed_layer(after_3d_ids)
    start_emb = embed_layer(torch.tensor([[model._start_3d_id]], device=device))
    end_emb = embed_layer(torch.tensor([[model._end_3d_id]], device=device))

    model_dtype = before_3d_embeds.dtype
    three_d_block = torch.cat(
        [start_emb.to(model_dtype), structure_proj.to(model_dtype), end_emb.to(model_dtype)], dim=1
    )
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
        next_embed = embed_layer(torch.tensor([[next_id]], dtype=torch.long, device=device)).to(
            current_embeds.dtype
        )
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        current_attention = torch.cat(
            [current_attention, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1
        )
        position_ids = torch.cat(
            [position_ids, torch.tensor([[position_ids.shape[1]]], device=device, dtype=torch.long)], dim=1
        )

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_batch_with_multi_token_structure(
    model,
    tokenizer: AutoTokenizer,
    batch_atoms: List[list],
    batch_coords: List,
    batch_user_contents: List[str],
    *,
    max_new_tokens: int = 512,
) -> List[str]:
    if not batch_atoms:
        return []

    device = next(model.llm.parameters()).device
    embed_layer = model.llm.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model.unimol.eval()
    atom_repr, atom_pad_mask = model._encode_atom_repr(batch_atoms, batch_coords, device)
    proj_dtype = next(model.multi_token_projection_layer.parameters()).dtype
    structure_proj = model.multi_token_projection_layer(
        atom_repr.to(dtype=proj_dtype), atom_pad_mask.to(device)
    ).to(dtype=embed_layer.weight.dtype)

    before_parts, after_parts = zip(*[split_chat_prefix(tokenizer, uc) for uc in batch_user_contents])
    before_ids_list = [tokenizer(b, return_tensors="pt").input_ids for b in before_parts]
    after_ids_list = [
        tokenizer(a, return_tensors="pt").input_ids if a else torch.zeros(1, 0, dtype=torch.long)
        for a in after_parts
    ]
    before_ids, before_mask = pad_token_ids(before_ids_list, pad_id, device)
    after_ids, after_mask = pad_token_ids(after_ids_list, pad_id, device)
    before_embeds = embed_layer(before_ids)
    after_embeds = embed_layer(after_ids)

    start_emb = embed_layer(torch.full((len(batch_atoms), 1), model._start_3d_id, dtype=torch.long, device=device))
    end_emb = embed_layer(torch.full((len(batch_atoms), 1), model._end_3d_id, dtype=torch.long, device=device))
    model_dtype = before_embeds.dtype
    three_d_block = torch.cat([start_emb.to(model_dtype), structure_proj.to(model_dtype), end_emb.to(model_dtype)], dim=1)
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
        gen_ids = row[prompt_len:] if row.shape[0] > prompt_len else row
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return texts


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
    mol_embeds = model.single_token_projection_layer(single_token_repr.to(dtype=proj_dtype)).unsqueeze(1)

    before_parts, after_parts = zip(*[split_chat_prefix(tokenizer, uc) for uc in batch_user_contents])
    before_ids_list = [tokenizer(b, return_tensors="pt").input_ids for b in before_parts]
    after_ids_list = [
        tokenizer(a, return_tensors="pt").input_ids if a else torch.zeros(1, 0, dtype=torch.long)
        for a in after_parts
    ]
    before_ids, before_mask = pad_token_ids(before_ids_list, pad_id, device)
    after_ids, after_mask = pad_token_ids(after_ids_list, pad_id, device)
    before_embeds = embed_layer(before_ids)
    after_embeds = embed_layer(after_ids)

    start_emb = embed_layer(torch.full((len(batch_atoms), 1), model._start_3d_id, dtype=torch.long, device=device))
    end_emb = embed_layer(torch.full((len(batch_atoms), 1), model._end_3d_id, dtype=torch.long, device=device))
    model_dtype = before_embeds.dtype
    three_d_block = torch.cat([start_emb.to(model_dtype), mol_embeds.to(model_dtype), end_emb.to(model_dtype)], dim=1)
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
        gen_ids = row[prompt_len:] if row.shape[0] > prompt_len else row
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return texts


def generate_description_batch(
    mode: str,
    model,
    tokenizer: AutoTokenizer,
    chunk: List[Dict[str, Any]],
    *,
    max_new_tokens: int,
) -> List[str]:
    user_contents = [s["user_content"] for s in chunk]
    if mode == "random_3d":
        return generate_batch_with_random_structure(
            model,
            tokenizer,
            user_contents,
            [int(s["sample_idx"]) for s in chunk],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    atoms = [s["atoms"] for s in chunk]
    coords = [s["coordinates"] for s in chunk]
    if mode == "multi_token":
        return generate_batch_with_multi_token_structure(
            model, tokenizer, atoms, coords, user_contents, max_new_tokens=max_new_tokens
        )
    return generate_batch_with_single_token_structure(
        model, tokenizer, atoms, coords, user_contents, max_new_tokens=max_new_tokens
    )


def generate_description_one(
    mode: str,
    model,
    tokenizer: AutoTokenizer,
    sample: Dict[str, Any],
    *,
    max_new_tokens: int,
) -> str:
    if mode == "random_3d":
        return generate_with_random_structure(
            model,
            tokenizer,
            sample["user_content"],
            sample_idx=int(sample["sample_idx"]),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if mode == "multi_token":
        return generate_with_multi_token_structure_greedy(
            model,
            tokenizer,
            sample["atoms"],
            sample["coordinates"],
            sample["user_content"],
            max_new_tokens=max_new_tokens,
        )
    return generate_with_single_token_structure(
        model,
        tokenizer,
        sample["atoms"],
        sample["coordinates"],
        sample["user_content"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def generate_property_prediction(
    mode: str,
    model,
    tokenizer,
    atoms,
    coords,
    user_content: str,
    *,
    max_new_tokens: int = 64,
    repetition_penalty: float = 1.05,
) -> str:
    if mode == "multi_token":
        return generate_with_multi_token_structure_greedy(
            model,
            tokenizer,
            atoms,
            coords,
            user_content,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    if mode == "random_3d":
        return generate_with_random_structure(
            model,
            tokenizer,
            user_content,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return generate_with_single_token_structure(
        model,
        tokenizer,
        atoms,
        coords,
        user_content,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


# ---------------------------------------------------------------------------
# Description multi-GPU batch
# ---------------------------------------------------------------------------


def atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def merge_partial_records(partial_paths: List[str]) -> List[dict]:
    merged: List[dict] = []
    for path in partial_paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                merged.extend(json.load(f))
        except json.JSONDecodeError:
            continue
    dedup: Dict[int, dict] = {int(rec["sample_idx"]): rec for rec in merged}
    return [dedup[k] for k in sorted(dedup)]


def _description_batch_worker(
    rank: int,
    world_size: int,
    gpu_ids: List[int],
    worker_args: dict,
    samples: list,
    partial_path: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[rank])
    torch.set_grad_enabled(False)

    shard = samples[rank::world_size]
    if not shard:
        atomic_write_json(partial_path, [])
        return

    tokenizer = AutoTokenizer.from_pretrained(worker_args["model_name"])
    model = load_description_model(
        worker_args["mode"],
        worker_args["model_name"],
        worker_args["three_d_encoder_dict"],
        worker_args["ckpt"],
        lora_r=worker_args["lora_r"],
        lora_alpha=worker_args["lora_alpha"],
        lora_target=worker_args["lora_target"],
        random_3d_seed=worker_args["random_3d_seed"],
    )
    model.eval()

    batch_size = worker_args["batch_size"]
    records = []
    t0 = time.perf_counter()
    print(f"[Worker {rank}] GPU={gpu_ids[rank]} | shard={len(shard)} | batch_size={batch_size}", flush=True)

    for start in range(0, len(shard), batch_size):
        chunk = shard[start : start + batch_size]
        preds = generate_description_batch(
            worker_args["mode"],
            model,
            tokenizer,
            chunk,
            max_new_tokens=worker_args["max_new_tokens"],
        )
        for sample, pred in zip(chunk, preds):
            records.append(
                {
                    "sample_idx": sample["sample_idx"],
                    "lmdb_path": sample["lmdb_path"],
                    "smiles": sample.get("smiles"),
                    "user_content": sample["user_content"],
                    "reference": sample["reference"],
                    "prediction": pred,
                }
            )
        atomic_write_json(partial_path, records)
        done = min(start + batch_size, len(shard))
        if worker_args["print_every"] > 0 and done % worker_args["print_every"] == 0:
            print(f"[Worker {rank}] {done}/{len(shard)}", flush=True)

    elapsed = time.perf_counter() - t0
    atomic_write_json(partial_path, records)
    print(
        f"[Worker {rank}] saved {len(records)} preds in {elapsed:.1f}s "
        f"({len(records) / elapsed:.2f} samp/s) -> {partial_path}",
        flush=True,
    )


def run_description_batch(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.init_ckpt):
        raise FileNotFoundError(f"--ckpt must be a directory: {args.init_ckpt}")

    test_lmdb = default_description_test_lmdb(args.test_lmdb)
    cfg = DESCRIPTION_MODE_CONFIG[args.mode]
    response_key = args.response_key or cfg.get("response_key") or RESPONSE_KEY
    samples = collect_description_samples(
        test_lmdb,
        include_smiles=cfg["include_smiles"],
        instruction=cfg["instruction"],
        response_key=response_key,
        max_samples=args.max_samples,
    )

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise RuntimeError("No GPUs specified via --gpus")

    print(
        f"[Infer] mode={args.mode} | samples={len(samples)} | gpus={gpu_ids} | "
        f"per_gpu_batch={args.batch_size} | ckpt={args.init_ckpt}"
    )

    save_json = os.path.abspath(args.save_json)
    live_json = (
        os.path.abspath(args.live_json)
        if args.live_json
        else os.path.join(os.path.expanduser("~"), "description_inference_live.json")
    )
    partial_dir = os.path.join(os.path.dirname(save_json) or ".", "_partial_multigpu")
    os.makedirs(partial_dir, exist_ok=True)
    partial_paths = [os.path.join(partial_dir, f"{args.mode}_rank{r}.json") for r in range(len(gpu_ids))]

    meta = {
        "mode": args.mode,
        "decode": "greedy",
        "ckpt": args.init_ckpt,
        "test_lmdb": test_lmdb,
        "response_key": response_key,
        "gpus": gpu_ids,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "n_total": len(samples),
    }
    worker_args = {
        "mode": args.mode,
        "model_name": args.model_name,
        "three_d_encoder_dict": args.three_d_encoder_dict,
        "ckpt": args.init_ckpt,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "print_every": args.print_every,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target": args.lora_target,
        "random_3d_seed": args.random_3d_seed,
    }

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    t0 = time.perf_counter()
    stop_event = threading.Event()

    def _live_loop():
        while True:
            records = merge_partial_records(partial_paths)
            elapsed = time.perf_counter() - t0
            status = "complete" if stop_event.is_set() else "in_progress"
            n_done, n_total = len(records), int(meta["n_total"])
            payload = {
                **meta,
                "status": status,
                "n_done": n_done,
                "n_total": n_total,
                "progress": round(n_done / n_total, 4) if n_total else 0.0,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "predictions": records,
                "wall_seconds": elapsed,
            }
            if records:
                payload["avg_ref_len"] = sum(len(r["reference"]) for r in records) / n_done
                payload["avg_pred_len"] = sum(len(r["prediction"]) for r in records) / n_done
            if elapsed > 0:
                payload["samples_per_second"] = n_done / elapsed
            atomic_write_json(save_json, payload)
            atomic_write_json(live_json, payload)
            if stop_event.is_set():
                break
            time.sleep(args.live_interval)

    live_thread = threading.Thread(target=_live_loop, daemon=True)
    live_thread.start()

    processes = []
    for rank in range(len(gpu_ids)):
        proc = mp.Process(
            target=_description_batch_worker,
            args=(rank, len(gpu_ids), gpu_ids, worker_args, samples, partial_paths[rank]),
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            stop_event.set()
            live_thread.join(timeout=10)
            raise RuntimeError(f"Worker failed with exit code {proc.exitcode}")

    stop_event.set()
    live_thread.join(timeout=30)

    merged = merge_partial_records(partial_paths)
    elapsed = time.perf_counter() - t0
    print("\n================ Multi-GPU Batch Inference ================")
    print(f"mode:           {args.mode}")
    print(f"total:          {len(merged)}")
    print(f"wall_time:      {elapsed:.1f}s ({len(merged) / elapsed:.2f} samp/s)")
    print(f"saved:          {save_json}")
    print("===========================================================")


# ---------------------------------------------------------------------------
# Stage 3 regression evaluation (property / Vaska / NiComplex)
# ---------------------------------------------------------------------------


def collect_regression_samples_from_records(
    records: Sequence[dict],
    task: str,
    *,
    mode: str = "single_token",
    instruction_override: Optional[str] = None,
) -> List[Tuple[list, np.ndarray, str, float]]:
    spec = get_task(task)
    rows: List[Tuple[list, np.ndarray, str, float]] = []
    for sample in records:
        try:
            y = float(sample[spec.target_key])
            if np.isnan(y):
                continue
        except (ValueError, TypeError, KeyError):
            continue
        if spec.requires_smiles_in_lmdb and not format_instruction_field(sample.get("smiles")):
            continue
        atoms, coords = atoms_coords_from_record(sample)
        user_content = resolve_user_content(
            task,
            sample,
            mode=mode,
            instruction_override=instruction_override,
        )
        rows.append((atoms, coords, user_content, y))
    if not rows:
        raise RuntimeError(f"No valid regression samples for task={task!r}")
    return rows


def _split_seed_for_task(args: argparse.Namespace, task: str) -> int:
    if args.split_seed is not None:
        return args.split_seed
    if task == "nicomplex_ddg":
        return NICOMPLEX_DEFAULTS["split_seed"]
    if task == "vaska_barrier":
        return VASKA_DEFAULTS["split_seed"]
    return VASKA_DEFAULTS["split_seed"]


def _random_80_10_10_test_split(samples: Sequence[dict], split_seed: int) -> List[dict]:
    """Same 80/10/10 shuffle as Stage3.py / Vaska training; return test fold only."""
    n_total = len(samples)
    indices = np.arange(n_total)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(indices)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    test_idx = indices[n_train + n_val :]
    return [samples[i] for i in test_idx]


def _run_regression_eval_loop(
    test_data: Sequence[Tuple[list, np.ndarray, str, float]],
    *,
    tag: str,
    mode: str,
    model,
    tokenizer,
    max_new_tokens: int,
    repetition_penalty: float,
    print_every: int,
) -> Tuple[np.ndarray, np.ndarray, int, List[dict]]:
    y_true, y_pred = [], []
    n_parse_fail = 0
    pred_records = []
    for i, (atoms, coords, user_content, y) in enumerate(test_data, start=1):
        out = generate_property_prediction(
            mode,
            model,
            tokenizer,
            atoms,
            coords,
            user_content,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        pred = parse_first_float(out)
        if pred is None or not np.isfinite(pred):
            n_parse_fail += 1
            pred_records.append({"ref": float(y), "pred_text": out, "pred_value": None})
        else:
            y_true.append(y)
            y_pred.append(pred)
            pred_records.append({"ref": float(y), "pred_text": out, "pred_value": float(pred)})
        if print_every > 0 and i % print_every == 0:
            print(f"[{tag}] {i}/{len(test_data)} | parsed={len(y_pred)} | parse_fail={n_parse_fail}")
    if not y_pred:
        raise RuntimeError(f"[{tag}] No valid numeric predictions parsed from model outputs.")
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64), n_parse_fail, pred_records


def _print_and_save_regression_metrics(
    args: argparse.Namespace,
    *,
    tag: str,
    task: str,
    unit: str,
    n_total: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_parse_fail: int,
    pred_records: List[dict],
    extra_meta: Optional[dict] = None,
) -> dict:
    mae_val = mae(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    print(f"\n================ Stage3 {tag} Metrics ================")
    print(f"task:           {task}")
    print(f"mode:           {args.mode}")
    print(f"unit:           {unit}")
    print(f"test_total:     {n_total}")
    print(f"pred_parsed:    {len(y_pred)}")
    print(f"parse_fail:     {n_parse_fail}")
    print(f"MAE:            {mae_val:.6f}")
    print(f"R2:             {r2_val:.6f}" if np.isfinite(r2_val) else "R2:             nan")
    print("=====================================================")

    payload = {
        "task": task,
        "mode": args.mode,
        "ckpt": args.stage3_ckpt,
        "unit": unit,
        "mae": mae_val,
        "r2": r2_val if np.isfinite(r2_val) else None,
        "n_total": n_total,
        "n_parsed": len(y_pred),
        "n_parse_fail": n_parse_fail,
        "predictions": pred_records,
    }
    if extra_meta:
        payload.update(extra_meta)

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[{tag}] Saved: {out_path}")
    return payload


def _load_stage3_eval_model(args: argparse.Namespace):
    if not os.path.isdir(args.stage3_ckpt):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.stage3_ckpt}")
    model = load_stage3_model(
        args.mode,
        args.model_name,
        args.three_d_encoder_dict,
        args.stage3_ckpt,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
        random_3d_seed=args.random_3d_seed,
    )
    model.eval()
    return model


def _eval_tmqm_task(args: argparse.Namespace, task: str) -> dict:
    from OOD.property.dataset_ood import TmQMgClusterSplitDataset

    if not args.split_csv and not getattr(args, "fixed_lmdb_eval", False):
        raise ValueError(
            f"Task {task!r} requires --split_csv for cluster OOD eval, "
            "or --fixed_lmdb_eval to evaluate on a held-out LMDB (--test_lmdb)."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = _load_stage3_eval_model(args)
    instruction_use = args.instruction or resolve_instruction(
        task,
        mode=args.mode if args.mode != "3d_only" else "3d_only",
        use_polished_description=args.use_polished_description,
    )

    if args.split_csv:
        lmdb_paths = args.lmdb_paths or DEFAULT_PROPERTY_OOD_LMDB
        if args.mode == "3d_only":
            raise ValueError("CSV-split property OOD eval supports single_token / multi_token modes only")
        dataset = TmQMgClusterSplitDataset(
            lmdb_paths,
            split_csv=args.split_csv,
            split_name=args.split_name,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            property_key=task,
            instruction=instruction_use,
            use_polished_description=args.use_polished_description,
        )
        test_data = collect_property_eval_samples(dataset)
        tag = f"stage3-{task}-csv-{args.split_name}"
        extra = {"split_csv": args.split_csv, "split_name": args.split_name, "lmdb_paths": lmdb_paths}
    elif getattr(args, "fixed_lmdb_eval", False):
        if not os.path.isfile(args.test_lmdb):
            raise FileNotFoundError(f"Test LMDB not found: {args.test_lmdb}")
        if args.mode == "3d_only":
            dataset = TmQMg3DOnlyUnimolDataset(
                [args.test_lmdb],
                tokenizer=tokenizer,
                max_samples=args.max_samples,
                property_key=task,
                instruction=instruction_use,
            )
        else:
            dataset = TmQMgSingleTokenUnimolDataset(
                [args.test_lmdb],
                tokenizer=tokenizer,
                max_samples=args.max_samples,
                property_key=task,
                instruction=instruction_use,
                use_polished_description=args.use_polished_description,
            )
        test_data = collect_property_eval_samples(dataset)
        tag = f"stage3-{task}"
        extra = {"test_lmdb": args.test_lmdb}
    else:
        raise AssertionError("unreachable: protocol validated above")

    print(f"[{tag}] valid={len(test_data)} | ckpt={args.stage3_ckpt}")
    y_true, y_pred, n_parse_fail, pred_records = _run_regression_eval_loop(
        test_data,
        tag=tag,
        mode=args.mode,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        print_every=args.print_every,
    )
    unit = PROPERTY_CONFIG[task].get("unit", "")
    return _print_and_save_regression_metrics(
        args,
        tag=tag,
        task=task,
        unit=unit,
        n_total=len(test_data),
        y_true=y_true,
        y_pred=y_pred,
        n_parse_fail=n_parse_fail,
        pred_records=pred_records,
        extra_meta=extra,
    )


def _eval_vaska_holdout(args: argparse.Namespace, holdout_ligand: str, all_raw: list) -> dict:
    from OOD.Vaska.ligand_split import lolo_split_by_ligand

    _, ood_raw = lolo_split_by_ligand(all_raw, holdout_ligand)
    test_records = filter_valid_vaska_records(ood_raw)
    test_data = collect_regression_samples_from_records(test_records, "vaska_barrier", mode="single_token")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = _load_stage3_eval_model(args)
    tag = f"stage3-vaska-{holdout_ligand}"
    print(f"[{tag}] ood_raw={len(ood_raw)} | valid={len(test_data)} | ckpt={args.stage3_ckpt}")

    y_true, y_pred, n_parse_fail, pred_records = _run_regression_eval_loop(
        test_data,
        tag=tag,
        mode="single_token",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        print_every=args.print_every,
    )
    return _print_and_save_regression_metrics(
        args,
        tag=tag,
        task="vaska_barrier",
        unit="kcal/mol",
        n_total=len(test_data),
        y_true=y_true,
        y_pred=y_pred,
        n_parse_fail=n_parse_fail,
        pred_records=pred_records,
        extra_meta={"holdout_ligand": holdout_ligand, "lmdb": args.lmdb},
    )


def _eval_vaska_random_split(args: argparse.Namespace, all_valid: list) -> dict:
    """Evaluate on 80/10/10 test fold (matches Stage3 vaska training split_seed)."""
    split_seed = _split_seed_for_task(args, "vaska_barrier")
    test_raw = _random_80_10_10_test_split(all_valid, split_seed)
    test_data = collect_regression_samples_from_records(test_raw, "vaska_barrier", mode="single_token")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = _load_stage3_eval_model(args)
    tag = f"stage3-vaska-seed_{split_seed}"
    print(
        f"[{tag}] total={len(all_valid)} | test_raw={len(test_raw)} | valid={len(test_data)} | "
        f"ckpt={args.stage3_ckpt}"
    )

    y_true, y_pred, n_parse_fail, pred_records = _run_regression_eval_loop(
        test_data,
        tag=tag,
        mode="single_token",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        print_every=args.print_every,
    )
    return _print_and_save_regression_metrics(
        args,
        tag=tag,
        task="vaska_barrier",
        unit="kcal/mol",
        n_total=len(test_data),
        y_true=y_true,
        y_pred=y_pred,
        n_parse_fail=n_parse_fail,
        pred_records=pred_records,
        extra_meta={"split_seed": split_seed, "lmdb": args.lmdb or DEFAULT_VASKA_LMDB},
    )


def _eval_nicomplex_random_split(args: argparse.Namespace, all_valid: list) -> dict:
    """Evaluate on 80/10/10 test fold (matches Stage3 nicomplex_ddg training split_seed)."""
    split_seed = _split_seed_for_task(args, "nicomplex_ddg")
    test_raw = _random_80_10_10_test_split(all_valid, split_seed)
    instruction = args.instruction or NI_INSTRUCTION
    test_data = collect_regression_samples_from_records(
        test_raw,
        "nicomplex_ddg",
        mode="single_token",
        instruction_override=instruction,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = _load_stage3_eval_model(args)
    tag = f"stage3-nicomplex-seed_{split_seed}"
    print(
        f"[{tag}] total={len(all_valid)} | test_raw={len(test_raw)} | valid={len(test_data)} | "
        f"ckpt={args.stage3_ckpt}"
    )

    y_true, y_pred, n_parse_fail, pred_records = _run_regression_eval_loop(
        test_data,
        tag=tag,
        mode="single_token",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        print_every=args.print_every,
    )
    return _print_and_save_regression_metrics(
        args,
        tag=tag,
        task="nicomplex_ddg",
        unit="kcal/mol",
        n_total=len(test_data),
        y_true=y_true,
        y_pred=y_pred,
        n_parse_fail=n_parse_fail,
        pred_records=pred_records,
        extra_meta={
            "split_seed": split_seed,
            "lmdb_paths": args.lmdb_paths or [DEFAULT_NICOMPLEX_LMDB],
        },
    )


def _eval_nicomplex_experiment(args: argparse.Namespace, experiment_name: str, all_valid: list) -> dict:
    from OOD.NiComplex.nicomplex_split import split_by_experiment

    _, ood_samples, spec = split_by_experiment(all_valid, experiment_name)
    instruction = args.instruction or NI_INSTRUCTION
    test_data = collect_regression_samples_from_records(
        ood_samples,
        "nicomplex_ddg",
        mode="single_token",
        instruction_override=instruction,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = _load_stage3_eval_model(args)
    tag = f"stage3-nicomplex-{experiment_name}"
    print(f"[{tag}] experiment={experiment_name} | valid={len(test_data)} | ckpt={args.stage3_ckpt}")

    y_true, y_pred, n_parse_fail, pred_records = _run_regression_eval_loop(
        test_data,
        tag=tag,
        mode="single_token",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        print_every=args.print_every,
    )
    return _print_and_save_regression_metrics(
        args,
        tag=tag,
        task="nicomplex_ddg",
        unit="kcal/mol",
        n_total=len(test_data),
        y_true=y_true,
        y_pred=y_pred,
        n_parse_fail=n_parse_fail,
        pred_records=pred_records,
        extra_meta={
            "ood_experiment": experiment_name,
            "split_col": spec["split_col"],
            "test_types": sorted(spec["test_types"]),
            "lmdb_paths": args.lmdb_paths or [DEFAULT_NICOMPLEX_LMDB],
        },
    )


def eval_stage3(args: argparse.Namespace) -> None:
    if getattr(args, "property", None):
        args.task = args.property
    task = normalize_task_name(args.task)
    if task not in STAGE3_TASKS:
        raise ValueError(f"Unknown task={task!r}; choose from {list(STAGE3_TASKS)}")

    if task in TMQM_TASKS:
        eval_stage3._last_summary = _eval_tmqm_task(args, task)  # type: ignore[attr-defined]
        return

    if task == "vaska_barrier":
        if args.mode != "single_token":
            raise ValueError("vaska_barrier inference supports --mode single_token only")
        from OOD.Vaska.ligand_split import LIGANDS

        lmdb_path = args.lmdb or DEFAULT_VASKA_LMDB

        if getattr(args, "random_split", False):
            all_valid, _ = load_merged_valid_vaska_records(lmdb_path, local_rank=0)
            if not all_valid:
                raise RuntimeError(f"Empty Vaska LMDB after filtering: {lmdb_path}")
            eval_stage3._last_summary = _eval_vaska_random_split(args, all_valid)  # type: ignore[attr-defined]
            return

        all_raw = read_vaska_lmdb(lmdb_path, max_samples=None, show_progress=True)
        if not all_raw:
            raise RuntimeError(f"Empty Vaska LMDB: {lmdb_path}")

        if args.run_all_ood:
            summaries = []
            for ligand in LIGANDS:
                fold_args = argparse.Namespace(**{**vars(args), "holdout_ligand": ligand})
                fold_args.save_json = args.save_json
                if args.save_json:
                    base, ext = os.path.splitext(args.save_json)
                    fold_args.save_json = f"{base}_{ligand}{ext or '.json'}"
                summaries.append(_eval_vaska_holdout(fold_args, ligand, all_raw))
            eval_stage3._last_summary = summaries  # type: ignore[attr-defined]
            return

        if args.holdout_ligand:
            eval_stage3._last_summary = _eval_vaska_holdout(args, args.holdout_ligand, all_raw)  # type: ignore[attr-defined]
            return
        raise ValueError(
            "vaska_barrier requires --holdout_ligand <name> or --run_all_ood for LOLO OOD eval. "
            "For random 80/10/10 test fold, pass --random_split (uses split_seed=43 by default)."
        )

    if task == "nicomplex_ddg":
        if args.mode != "single_token":
            raise ValueError("nicomplex_ddg inference supports --mode single_token only")
        from OOD.NiComplex.nicomplex_split import EXPERIMENT_NAMES

        lmdb_paths = args.lmdb_paths or ([args.lmdb] if args.lmdb else [DEFAULT_NICOMPLEX_LMDB])
        all_valid, _ = load_merged_valid_nicomplex_records(lmdb_paths, local_rank=0)
        if not all_valid:
            raise RuntimeError(f"Empty NiComplex LMDB: {lmdb_paths}")

        if args.run_all_ood:
            summaries = []
            for experiment in EXPERIMENT_NAMES:
                fold_args = argparse.Namespace(**{**vars(args), "ood_experiment": experiment})
                if args.save_json:
                    base, ext = os.path.splitext(args.save_json)
                    fold_args.save_json = f"{base}_{experiment}{ext or '.json'}"
                else:
                    fold_args.save_json = None
                summaries.append(_eval_nicomplex_experiment(fold_args, experiment, all_valid))
            eval_stage3._last_summary = summaries  # type: ignore[attr-defined]
            return

        if args.ood_experiment:
            eval_stage3._last_summary = _eval_nicomplex_experiment(args, args.ood_experiment, all_valid)  # type: ignore[attr-defined]
            return
        if getattr(args, "random_split", False):
            eval_stage3._last_summary = _eval_nicomplex_random_split(args, all_valid)  # type: ignore[attr-defined]
            return
        raise ValueError(
            "nicomplex_ddg requires --ood_experiment <name> or --run_all_ood for scaffold OOD eval. "
            "For random 80/10/10 test fold, pass --random_split (uses split_seed=42 by default)."
        )

    raise ValueError(f"Unhandled task={task!r}")


def eval_property_lmdb(args: argparse.Namespace) -> None:
    """Backward-compatible wrapper around ``eval_stage3`` for TmQM properties."""
    if not hasattr(args, "task") or not args.task:
        args.task = args.property
    eval_stage3(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_description_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "description",
        help="Multi-GPU batched description inference (Stage2 default + ablations)",
    )
    p.add_argument("--mode", required=True, choices=list(DESCRIPTION_MODES))
    p.add_argument("--model_name", default=DESCRIPTION_DEFAULTS["model_name"])
    p.add_argument("--3D_encoder_dict", dest="three_d_encoder_dict", default=DESCRIPTION_DEFAULTS["3D_encoder_dict"])
    p.add_argument("--ckpt", dest="init_ckpt", required=True)
    p.add_argument("--test_lmdb", nargs="+", default=None)
    p.add_argument(
        "--response_key",
        default=None,
        help=f"LMDB reference field (default: {STAGE2_RESPONSE_KEY} for stage2, {RESPONSE_KEY} for ablations)",
    )
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--print_every", type=int, default=50)
    p.add_argument("--save_json", required=True)
    p.add_argument("--live_json", default=None)
    p.add_argument("--live_interval", type=float, default=5.0)
    p.add_argument("--random_3d_seed", type=int, default=DESCRIPTION_DEFAULTS["random_3d_seed"])
    p.add_argument("--lora_r", type=int, default=DESCRIPTION_DEFAULTS["lora_r"])
    p.add_argument("--lora_alpha", type=int, default=DESCRIPTION_DEFAULTS["lora_alpha"])
    p.add_argument("--lora_target", default=DESCRIPTION_DEFAULTS["lora_target"])


def _add_stage3_regression_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", default="single_token", choices=list(STAGE3_INFER_MODES))
    p.add_argument("--model_name", default=VASKA_DEFAULTS["model_name"])
    p.add_argument("--3D_encoder_dict", dest="three_d_encoder_dict", default=VASKA_DEFAULTS["3D_encoder_dict"])
    p.add_argument("--Stage3_ckpt", dest="stage3_ckpt", required=True)
    p.add_argument("--test_lmdb", default=DEFAULT_PROPERTY_TEST_LMDB)
    p.add_argument(
        "--lmdb_paths",
        nargs="+",
        default=None,
        help="Property OOD: one or more LMDB files (CSD keys); filtered by --split_csv",
    )
    p.add_argument("--lmdb", default=None, help="Vaska LMDB path")
    p.add_argument(
        "--split_csv",
        default=None,
        help="Property OOD: CSV with 'CSD code' + 'split', or directory with train.csv / test.csv",
    )
    p.add_argument("--split_name", default="test", choices=["train", "test"])
    p.add_argument(
        "--fixed_lmdb_eval",
        action="store_true",
        help="TmQM property: evaluate on --test_lmdb (fixed held-out LMDB), not cluster OOD CSV",
    )
    p.add_argument(
        "--holdout_ligand",
        default=None,
        help="Vaska 26-ligand OOD: held-out ligand (see OOD.Vaska.ligand_split.LIGANDS)",
    )
    p.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Random 80/10/10 test fold seed (default: 42 for nicomplex_ddg, 43 for vaska_barrier)",
    )
    p.add_argument(
        "--random_split",
        action="store_true",
        help="Evaluate random 80/10/10 test fold (Vaska / NiComplex) instead of OOD holdout",
    )
    p.add_argument(
        "--ood_experiment",
        "--experiment",
        dest="ood_experiment",
        default=None,
        choices=[
            "train_rest_test_Pybox",
            "train_rest_test_Biox",
            "train_rest_test_Biim",
        ],
        help="NiComplex scaffold OOD experiment",
    )
    p.add_argument(
        "--run_all_ood",
        action="store_true",
        help="Run all Vaska ligand folds or all NiComplex scaffold experiments",
    )
    p.add_argument("--instruction", default=None, help="Override task instruction")
    p.add_argument(
        "--use_polished_description",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="TmQM homo_lumo_gap: prepend polished_description",
    )
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_target", default="all", choices=["qv", "qkv", "all"])
    p.add_argument("--random_3d_seed", type=int, default=DESCRIPTION_DEFAULTS["random_3d_seed"])
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--print_every", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--save_json", default=None)


def _add_property_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("property", help="(Legacy) TmQM property eval — prefer: inference.py stage3")
    p.add_argument("--property", default="dipole_moment", choices=list(PROPERTY_CONFIG.keys()))
    p.add_argument("--task", default=None, choices=list(STAGE3_TASKS))
    _add_stage3_regression_args(p)


def _add_stage3_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("stage3", help="Stage 3 regression eval (property / Vaska / NiComplex)")
    p.add_argument("--task", required=True, choices=list(STAGE3_TASKS))
    p.add_argument("--property", default=None, choices=list(TMQM_TASKS), help="Deprecated alias for --task")
    _add_stage3_regression_args(p)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Unified inference for description and Stage 3 tasks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_description_parser(sub)
    _add_property_parser(sub)
    _add_stage3_parser(sub)
    args = parser.parse_args(argv)
    torch.set_grad_enabled(False)
    if args.cmd == "description":
        run_description_batch(args)
    elif args.cmd in ("property", "stage3"):
        eval_stage3(args)


if __name__ == "__main__":
    main()
