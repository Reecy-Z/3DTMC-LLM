"""
Shared utilities for Uni-Mol BOS + Qwen chat templates with an object_ref (3D) slot.

Used by Property / Stage1 / NiComplex / Vaska / Stage2: LMDB IO, geometry prep,
Uni-Mol batching, BosProjection embed fusion, and chat tokenization around``<|im_end|>`` (object_ref boundary).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import lmdb
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import distance_matrix
from transformers import Trainer
from tqdm import tqdm

TMC_LLM_ROOT = os.path.dirname(os.path.abspath(__file__))
UNIMOL_ORIGIN = "/home/zhujingyuan/Uni-Mol"

MAX_SEQ_LENGTH = 512
ATOM_DIM = 512
UNIMOL_MAX_SEQ_LEN = 512

# Qwen3 chat template: 3D placeholder splits here (must match training scripts).
OBJECT_REF_CHAT_SEP = "<|im_end|>"


def ensure_unimol_import_paths() -> None:
    """Prepend local Uni-Core / Uni-Mol paths so ``import unimol`` works."""
    for p in [
        TMC_LLM_ROOT,
        os.path.join(TMC_LLM_ROOT, "Uni-Core"),
        UNIMOL_ORIGIN,
        os.path.join(UNIMOL_ORIGIN, "Uni-Core"),
    ]:
        if p not in sys.path and os.path.isdir(p):
            sys.path.insert(0, p)
    if TMC_LLM_ROOT not in sys.path:
        sys.path.insert(0, TMC_LLM_ROOT)


def _lmdb_env_kwargs():
    return dict(subdir=False, readonly=True, lock=False, readahead=True, meminit=False, max_readers=256)


def read_lmdb(lmdb_path, max_samples=None, show_progress=True):
    env = lmdb.open(lmdb_path, **_lmdb_env_kwargs())
    txn = env.begin()
    cursor = txn.cursor()
    data_list = []
    total = min(max_samples, env.stat()["entries"]) if max_samples else env.stat()["entries"]
    desc = f"LMDB {os.path.basename(lmdb_path)}"
    it = tqdm(cursor, total=total, desc=desc, unit="samples", mininterval=1.0) if (
        show_progress and int(os.environ.get("LOCAL_RANK", 0)) == 0
    ) else cursor
    for count, (_, value) in enumerate(it):
        data_list.append(pickle.loads(value))
        if max_samples and (count + 1) >= max_samples:
            break
    env.close()
    return data_list


def _atoms_coords_remove_h_center(atoms, coords):
    atoms = [str(a).strip() for a in atoms]
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim == 3:
        coords = coords[0]
    if len(atoms) != coords.shape[0]:
        return atoms, coords
    mask = np.array([a.upper() != "H" and a != "1" for a in atoms], dtype=bool)
    atoms = [a for a, m in zip(atoms, mask) if m]
    coords = coords[mask]
    if coords.size > 0:
        coords = coords - coords.mean(axis=0)
    return atoms, coords


def format_instruction_field(x) -> str:
    """Normalize values for ``instruction.format(...)`` (SMILES, temp, etc.)."""
    if x is None:
        return ""
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return ""
        if float(x).is_integer():
            return str(int(x))
        return str(float(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x).strip()


def build_batch_multi(
    list_atoms,
    list_coordinates,
    dictionary,
    max_seq_len=512,
    pad_idx=0,
    bos_idx=1,
    eos_idx=2,
    device="cpu",
):
    num_types = len(dictionary)
    batch_seq_lens = [1 + len(atoms) + 1 for atoms in list_atoms]
    L = min(max(batch_seq_lens), max_seq_len)
    list_src_tokens, list_src_coord, list_src_distance, list_src_edge_type = [], [], [], []
    for atoms, coordinates in zip(list_atoms, list_coordinates):
        token_ids = [dictionary.index(a) if a in dictionary.indices else dictionary.unk() for a in atoms]
        seq = [bos_idx] + token_ids + [eos_idx]
        pad_len = max(0, L - len(seq))
        src_tokens = torch.tensor([seq], dtype=torch.long)
        src_tokens = nn.functional.pad(src_tokens, (0, pad_len), value=pad_idx)
        coords = np.asarray(coordinates, dtype=np.float32)
        coord_bos = np.zeros((1, 3), dtype=np.float32)
        coord_eos = np.zeros((1, 3), dtype=np.float32)
        coord_pad = np.zeros((pad_len, 3), dtype=np.float32)
        full_coord = np.vstack([coord_bos, coords, coord_eos, coord_pad])
        src_coord = torch.from_numpy(full_coord).unsqueeze(0)
        dist = distance_matrix(full_coord, full_coord).astype(np.float32)
        src_distance = torch.from_numpy(dist).unsqueeze(0)
        tokens_np = src_tokens[0].numpy()
        ei, ej = tokens_np.reshape(-1, 1), tokens_np.reshape(1, -1)
        src_edge_type = torch.from_numpy((ei * num_types + ej).astype(np.int64)).unsqueeze(0)
        list_src_tokens.append(src_tokens)
        list_src_coord.append(src_coord)
        list_src_distance.append(src_distance)
        list_src_edge_type.append(src_edge_type)
    return {
        "src_tokens": torch.cat(list_src_tokens, dim=0).to(device),
        "src_distance": torch.cat(list_src_distance, dim=0).to(device),
        "src_coord": torch.cat(list_src_coord, dim=0).to(device),
        "src_edge_type": torch.cat(list_src_edge_type, dim=0).to(device),
    }


def extract_bos_repr(encoder_rep):
    return encoder_rep[:, 0, :]


class BosProjectionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.projection = nn.Linear(ATOM_DIM, hidden_size)

    def forward(self, x):
        return self.projection(x)


def tokenize_generation_sample_object_ref(
    tokenizer,
    user_content: str,
    assistant_content: str,
    *,
    sep: str = OBJECT_REF_CHAT_SEP,
    prefix_len_mode: str = "min",
    tokenizer_full: Optional[Dict[str, Any]] = None,
    tokenizer_prefix: Optional[Dict[str, Any]] = None,
    tokenizer_split_parts: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[int]]:
    """
    Build before_3d / after_3d / response token ids for supervised generation with a 3D slot.

    ``tokenizer_*`` are optional kwargs forwarded to ``tokenizer(...)`` for the full sequence,
    the user prefix only, and the before/after_3d re-encodes respectively.

    ``prefix_len_mode``: ``\"min\"`` (Property / NiComplex / Vaska) or ``\"prefix\"`` (legacy Stage1:
    ``len(prefix_enc)`` only).
    """
    tokenizer_full = dict(tokenizer_full or {})
    tokenizer_prefix = dict(tokenizer_prefix or {})
    tokenizer_split_parts = dict(tokenizer_split_parts or {})

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(full_text, return_tensors=None, **tokenizer_full)
    prefix_enc = tokenizer(prefix_str, return_tensors=None, **tokenizer_prefix)
    if prefix_len_mode == "min":
        prefix_len = min(len(prefix_enc["input_ids"]), len(enc["input_ids"]))
    elif prefix_len_mode == "prefix":
        prefix_len = len(prefix_enc["input_ids"])
    else:
        raise ValueError(f"prefix_len_mode must be 'min' or 'prefix', got {prefix_len_mode!r}")
    response_ids = enc["input_ids"][prefix_len:]
    if len(response_ids) == 0:
        raise RuntimeError(
            "response_ids is empty: check assistant text and that tokenizer/chat_template matches full_text."
        )
    before_3d_str, after_3d_str = prefix_str.split(sep, 1)
    after_3d_str = sep + after_3d_str
    before_3d_enc = tokenizer(before_3d_str, return_tensors=None, **tokenizer_split_parts)
    after_3d_enc = tokenizer(after_3d_str, return_tensors=None, **tokenizer_split_parts)
    return {
        "before_3d_ids": before_3d_enc["input_ids"],
        "after_3d_ids": after_3d_enc["input_ids"],
        "response_ids": response_ids,
    }


def _pad_ids(id_lists, pad_id, dtype=torch.long):
    max_len = max(len(ids) for ids in id_lists)
    B = len(id_lists)
    padded = torch.full((B, max_len), pad_id, dtype=dtype)
    mask = torch.zeros(B, max_len, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        L = len(ids)
        padded[i, :L] = torch.tensor(ids, dtype=dtype)
        mask[i, :L] = 1
    return padded, mask


class BridgeUnimolCollator:
    """Pads chat prefix slices + response + structure fields for BOS-slot models."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        before_3d_ids = [b["before_3d_ids"] for b in batch]
        after_3d_ids = [b["after_3d_ids"] for b in batch]
        list_atoms = [b["atoms"] for b in batch]
        list_coordinates = [b["coordinates"] for b in batch]
        before_3d_padded, before_3d_mask = _pad_ids(before_3d_ids, self.pad_id)
        after_3d_padded, after_3d_mask = _pad_ids(after_3d_ids, self.pad_id)
        response_ids = [b["response_ids"] for b in batch]
        response_padded, response_mask = _pad_ids(response_ids, self.pad_id)
        return {
            "before_3d_ids": before_3d_padded,
            "before_3d_mask": before_3d_mask,
            "after_3d_ids": after_3d_padded,
            "after_3d_mask": after_3d_mask,
            "response_ids": response_padded,
            "response_mask": response_mask,
            "list_atoms": list_atoms,
            "list_coordinates": list_coordinates,
            "labels": response_padded,
        }


def _pad_stack_embeds(emb_list, mask_list, label_list, device, dtype_emb):
    max_len = max(e.shape[1] for e in emb_list)
    H = emb_list[0].shape[2]
    pad_emb = torch.zeros(1, 1, H, device=device, dtype=dtype_emb)
    out_emb, out_mask, out_labels = [], [], []
    for e, m, lbl in zip(emb_list, mask_list, label_list):
        pad_len = max_len - e.shape[1]
        out_emb.append(torch.cat([e, pad_emb.expand(1, pad_len, -1)], dim=1))
        out_mask.append(torch.cat([m, torch.zeros(1, pad_len, dtype=m.dtype, device=device)], dim=1))
        out_labels.append(torch.cat([lbl, torch.full((1, pad_len), -100, dtype=torch.long, device=device)], dim=1))
    return torch.cat(out_emb, dim=0), torch.cat(out_mask, dim=0), torch.cat(out_labels, dim=0)


def build_embeds_text_only(llm, device, before_3d_ids, before_3d_mask, response_ids, response_mask):
    """User prefix + assistant response embeddings only (no 3D / object_ref slot)."""
    embed_fn = llm.get_input_embeddings()
    before_3d_ids = before_3d_ids.to(device)
    response_ids = response_ids.to(device)
    before_3d_mask = before_3d_mask.to(device)
    response_mask = response_mask.to(device)
    before_3d_embeds = embed_fn(before_3d_ids)
    response_embeds = embed_fn(response_ids)
    L1, Lr = before_3d_mask.sum(dim=1), response_mask.sum(dim=1)
    emb_list, mask_list, label_list = [], [], []
    for i in range(before_3d_ids.shape[0]):
        l1, lr = L1[i].item(), Lr[i].item()
        fused = torch.cat([before_3d_embeds[i : i + 1, :l1], response_embeds[i : i + 1, :lr]], dim=1)
        fused_mask = torch.ones(1, fused.shape[1], dtype=torch.long, device=device)
        fused_labels = torch.full((1, l1), -100, dtype=torch.long, device=device)
        resp_labels = response_ids[i : i + 1, :lr].clone()
        resp_labels[response_mask[i : i + 1, :lr] == 0] = -100
        all_labels = torch.cat([fused_labels, resp_labels], dim=1)
        emb_list.append(fused)
        mask_list.append(fused_mask)
        label_list.append(all_labels)
    return _pad_stack_embeds(emb_list, mask_list, label_list, device=device, dtype_emb=before_3d_embeds.dtype)


def build_embeds_bos_only(
    llm,
    device,
    before_3d_ids,
    before_3d_mask,
    after_3d_ids,
    after_3d_mask,
    response_ids,
    response_mask,
    bos_repr,
    bos_projection_layer,
    start_3d_id,
    end_3d_id,
):
    """Insert a single projected BOS vector between object_ref start/end token embeddings."""
    embed_fn = llm.get_input_embeddings()
    B = before_3d_ids.shape[0]
    before_3d_ids = before_3d_ids.to(device)
    after_3d_ids = after_3d_ids.to(device)
    response_ids = response_ids.to(device)
    before_3d_mask = before_3d_mask.to(device)
    after_3d_mask = after_3d_mask.to(device)
    response_mask = response_mask.to(device)
    before_3d_embeds = embed_fn(before_3d_ids)
    after_3d_embeds = embed_fn(after_3d_ids)
    response_embeds = embed_fn(response_ids)
    start_emb = embed_fn(torch.tensor([[start_3d_id]], device=device))
    end_emb = embed_fn(torch.tensor([[end_3d_id]], device=device))
    dtype_emb = before_3d_embeds.dtype

    proj_dtype = next(bos_projection_layer.parameters()).dtype
    bos_repr = bos_repr.to(device=device, dtype=proj_dtype)
    bos_proj = bos_projection_layer(bos_repr).unsqueeze(1).to(dtype_emb)

    L1, L2, Lr = before_3d_mask.sum(dim=1), after_3d_mask.sum(dim=1), response_mask.sum(dim=1)
    emb_list, mask_list, label_list = [], [], []
    for i in range(B):
        l1, l2, lr = L1[i].item(), L2[i].item(), Lr[i].item()
        if lr == 0:
            raise RuntimeError(
                "response effective length is 0 (Lr=0): labels would be prefix -100 only and CE loss is NaN. "
                "Check Dataset for empty response_ids or padding mask issues."
            )
        mol_emb = bos_proj[i : i + 1]
        fused = torch.cat([
            before_3d_embeds[i : i + 1, :l1],
            start_emb.expand(1, -1, -1),
            mol_emb,
            end_emb.expand(1, -1, -1),
            after_3d_embeds[i : i + 1, :l2],
        ], dim=1)
        fused_mask = torch.ones(1, fused.shape[1], dtype=torch.long, device=device)
        fused_labels = torch.full((1, fused.shape[1]), -100, dtype=torch.long, device=device)
        all_emb = torch.cat([fused, response_embeds[i : i + 1, :lr]], dim=1)
        all_mask = torch.cat([fused_mask, response_mask[i : i + 1, :lr]], dim=1)
        resp_labels = response_ids[i : i + 1, :lr].clone()
        resp_labels[response_mask[i : i + 1, :lr] == 0] = -100
        all_labels = torch.cat([fused_labels, resp_labels], dim=1)
        emb_list.append(all_emb)
        mask_list.append(all_mask)
        label_list.append(all_labels)
    return _pad_stack_embeds(emb_list, mask_list, label_list, device=device, dtype_emb=dtype_emb)


def unwrap_hf_model(model):
    m = model
    while hasattr(m, "module"):
        m = m.module
    return m


class BridgeUnimolFullTrainer(Trainer):
    """Saves LoRA adapter, unimol.pt, bos_projection.pt, and tokenizer (BOS-only pipeline)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError(
                "compute_loss: outputs.loss is None — check that labels are passed in training."
            )

        if loss.dim() > 0:
            loss = loss.mean()

        if not torch.isfinite(loss):
            try:
                lv = float(loss.detach().cpu().item())
            except Exception:
                lv = float("nan")
            raise RuntimeError(
                f"Training loss is not finite (loss={lv}); stopping. "
                "Check data (empty response, labels all -100), LR, and mixed precision."
            )

        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)
        model = unwrap_hf_model(self.model)
        model.llm.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
        if model.bos_projection_layer is not None:
            bos_proj_state = {k: v.cpu() for k, v in model.bos_projection_layer.state_dict().items()}
            torch.save(bos_proj_state, os.path.join(output_dir, "bos_projection.pt"))
        torch.save({"model": model.unimol.state_dict()}, os.path.join(output_dir, "unimol.pt"))
        # inference_dipole_bridge_unimol_full.load_model_full reads bos_only / include_bos
        with open(os.path.join(output_dir, "bridge_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "atom_dim": ATOM_DIM,
                    "include_bos": True,
                    "bos_only": True,
                },
                f,
                indent=2,
            )


ensure_unimol_import_paths()
