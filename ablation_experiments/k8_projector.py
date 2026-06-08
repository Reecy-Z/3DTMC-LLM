"""8 learnable query tokens + cross-attention over Uni-Mol atom hidden states (3D-MoLM-style K=8)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

NUM_STRUCTURE_QUERIES = 8
K8_PROJECTION_PT = "k8_projection.pt"


def extract_atom_repr_padded(
    encoder_rep: torch.Tensor,
    list_atoms: Sequence[Sequence],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Atom positions ``encoder_rep[:, 1:1+n_atoms]`` padded to batch max length.

    Returns:
        atom_repr: [B, L_max, D]
        atom_pad_mask: [B, L_max], True = padding (ignored in attention)
    """
    B, _, D = encoder_rep.shape
    n_atoms_list = [len(atoms) for atoms in list_atoms]
    max_n = max(n_atoms_list) if n_atoms_list else 0
    if max_n == 0:
        return (
            torch.zeros(B, 0, D, device=encoder_rep.device, dtype=encoder_rep.dtype),
            torch.ones(B, 0, dtype=torch.bool, device=encoder_rep.device),
        )
    atom_repr = torch.zeros(B, max_n, D, device=encoder_rep.device, dtype=encoder_rep.dtype)
    atom_pad_mask = torch.ones(B, max_n, dtype=torch.bool, device=encoder_rep.device)
    for i, n in enumerate(n_atoms_list):
        if n <= 0:
            continue
        atom_repr[i, :n] = encoder_rep[i, 1 : 1 + n]
        atom_pad_mask[i, :n] = False
    return atom_repr, atom_pad_mask


class K8QueryProjector(nn.Module):
    """Map variable-length atom encoder states to K fixed LLM embeddings via learnable queries."""

    def __init__(
        self,
        atom_dim: int,
        hidden_size: int,
        num_queries: int = NUM_STRUCTURE_QUERIES,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, atom_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.cross_attn = nn.MultiheadAttention(atom_dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(atom_dim)
        self.norm_kv = nn.LayerNorm(atom_dim)
        self.norm_ffn = nn.LayerNorm(atom_dim)
        self.ffn = nn.Sequential(
            nn.Linear(atom_dim, atom_dim * 4),
            nn.GELU(),
            nn.Linear(atom_dim * 4, atom_dim),
        )
        self.to_llm = nn.Linear(atom_dim, hidden_size)

    def forward(self, atom_repr: torch.Tensor, atom_pad_mask: torch.Tensor) -> torch.Tensor:
        """Returns projected structure tokens [B, K, hidden_size]."""
        if atom_repr.shape[1] == 0:
            raise RuntimeError("K8QueryProjector: no atom tokens in batch (empty structures).")
        B = atom_repr.shape[0]
        q = self.query_tokens.expand(B, -1, -1)
        kv = self.norm_kv(atom_repr)
        qn = self.norm_q(q)
        attn_out, _ = self.cross_attn(
            qn,
            kv,
            kv,
            key_padding_mask=atom_pad_mask,
            need_weights=False,
        )
        x = q + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return self.to_llm(x)


def build_embeds_k8(
    llm,
    device,
    before_3d_ids,
    before_3d_mask,
    after_3d_ids,
    after_3d_mask,
    response_ids,
    response_mask,
    atom_repr,
    atom_pad_mask,
    k8_projection_layer: K8QueryProjector,
    start_3d_id,
    end_3d_id,
):
    """Insert K projected structure vectors between object_ref start/end token embeddings."""
    from utils import _pad_stack_embeds

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

    proj_dtype = next(k8_projection_layer.parameters()).dtype
    atom_repr = atom_repr.to(device=device, dtype=proj_dtype)
    structure_proj = k8_projection_layer(atom_repr, atom_pad_mask.to(device)).to(dtype=dtype_emb)

    L1, L2, Lr = before_3d_mask.sum(dim=1), after_3d_mask.sum(dim=1), response_mask.sum(dim=1)
    emb_list, mask_list, label_list = [], [], []
    for i in range(B):
        l1, l2, lr = L1[i].item(), L2[i].item(), Lr[i].item()
        if lr == 0:
            raise RuntimeError(
                "response effective length is 0 (Lr=0): labels would be prefix -100 only and CE loss is NaN."
            )
        struct_emb = structure_proj[i : i + 1]
        fused = torch.cat(
            [
                before_3d_embeds[i : i + 1, :l1],
                start_emb.expand(1, -1, -1),
                struct_emb,
                end_emb.expand(1, -1, -1),
                after_3d_embeds[i : i + 1, :l2],
            ],
            dim=1,
        )
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
