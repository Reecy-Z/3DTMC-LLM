"""Stage3 multimodal model with K=8 learnable structure query tokens (ablation vs single-token)."""
from __future__ import annotations

import json
import os

import torch

from multimodal_LLM import RECIPE_STAGE3, MultimodalModel
from utils import (
    ATOM_DIM,
    THREE_D_ENCODER_STATE_PT,
    UNIMOL_MAX_SEQ_LEN,
    MultimodalFullTrainer,
    build_batch_multi,
    unwrap_hf_model,
)

from ablation_experiments.k8_projector import (
    K8_PROJECTION_PT,
    K8QueryProjector,
    NUM_STRUCTURE_QUERIES,
    build_embeds_k8,
    extract_atom_repr_padded,
)


class MultimodalModelK8(MultimodalModel):
    """Same stack as MultimodalModel stage3, but 8 query tokens attend to Uni-Mol atom hiddens."""

    def __init__(
        self,
        *args,
        load_pretrained_projection: bool = False,
        load_pretrained_lora: bool = False,
        **kwargs,
    ):
        kwargs["load_pretrained_projection"] = False
        kwargs["load_pretrained_lora"] = load_pretrained_lora
        init_ckpt = kwargs.get("init_ckpt")
        super().__init__(*args, **kwargs)
        if self.recipe != RECIPE_STAGE3:
            raise ValueError("MultimodalModelK8 only supports recipe=stage3_full_sft")
        hidden_size = self.single_token_projection_layer.projection.out_features
        device = next(self.llm.parameters()).device
        dtype = next(self.single_token_projection_layer.parameters()).dtype
        self.k8_projection_layer = K8QueryProjector(ATOM_DIM, hidden_size, num_queries=NUM_STRUCTURE_QUERIES).to(
            device=device, dtype=dtype
        )
        self.single_token_projection_layer = None
        train_projection = self._train_projection if self._train_projection is not None else True
        for p in self.k8_projection_layer.parameters():
            p.requires_grad = train_projection
        if init_ckpt and os.path.isdir(init_ckpt):
            k8_path = os.path.join(init_ckpt, K8_PROJECTION_PT)
            if os.path.isfile(k8_path):
                state = torch.load(k8_path, map_location="cpu", weights_only=True)
                self.k8_projection_layer.load_state_dict(state, strict=True)
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[3DTMC-LLM-K8] Loaded k8 projection from {k8_path}")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            lora_note = "Stage2 LoRA" if self._load_pretrained_lora else "new LoRA on base Qwen (from scratch)"
            print(
                f"[3DTMC-LLM-K8] Uni-Mol atom tokens -> {NUM_STRUCTURE_QUERIES} learnable queries "
                f"(cross-attn) -> LLM ({ATOM_DIM}->{hidden_size}); {lora_note}"
            )

    def _encode_atom_repr(self, list_atoms, list_coordinates, device):
        batch_dict = build_batch_multi(
            list_atoms,
            list_coordinates,
            self.dictionary,
            max_seq_len=UNIMOL_MAX_SEQ_LEN,
            pad_idx=self._pad_idx,
            single_token_idx=self._single_token_idx,
            eos_idx=self._eos_idx,
            device=str(device),
        )
        encoder_rep, _ = self.unimol(
            batch_dict["src_tokens"],
            batch_dict["src_distance"],
            batch_dict["src_coord"],
            batch_dict["src_edge_type"],
        )
        return extract_atom_repr_padded(encoder_rep, list_atoms)

    def forward(
        self,
        before_3d_ids,
        before_3d_mask,
        after_3d_ids,
        after_3d_mask,
        response_ids,
        response_mask,
        list_atoms,
        list_coordinates,
        sample_types=None,
        **kwargs,
    ):
        device = next(self.llm.parameters()).device
        atom_repr, atom_pad_mask = self._encode_atom_repr(list_atoms, list_coordinates, device)
        inputs_embeds, attention_mask, labels = build_embeds_k8(
            self.llm,
            device,
            before_3d_ids,
            before_3d_mask,
            after_3d_ids,
            after_3d_mask,
            response_ids,
            response_mask,
            atom_repr,
            atom_pad_mask,
            self.k8_projection_layer,
            self._start_3d_id,
            self._end_3d_id,
        )
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)


class MultimodalK8Trainer(MultimodalFullTrainer):
    """Saves k8_projection.pt instead of single_token_projection.pt."""

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)
        model = unwrap_hf_model(self.model)
        model.llm.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
        if getattr(model, "k8_projection_layer", None) is not None:
            state = {k: v.cpu() for k, v in model.k8_projection_layer.state_dict().items()}
            torch.save(state, os.path.join(output_dir, K8_PROJECTION_PT))
        torch.save({"model": model.unimol.state_dict()}, os.path.join(output_dir, THREE_D_ENCODER_STATE_PT))
        with open(os.path.join(output_dir, "multimodal_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "atom_dim": ATOM_DIM,
                    "structure_tokens": "k8_query",
                    "num_structure_queries": NUM_STRUCTURE_QUERIES,
                    "include_single_token": False,
                    "single_token_only": False,
                    "load_pretrained_lora": getattr(model, "_load_pretrained_lora", False),
                },
                f,
                indent=2,
            )
