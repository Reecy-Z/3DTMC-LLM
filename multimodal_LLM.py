"""
Unified 3D encoder single-token + Qwen3 (4-bit) stack for Stage1 / Stage2 / downstream full SFT.

Recipes (string values):
  - stage1_frozen_llm: frozen LLM; train 3D encoder + single-token projection only.
  - stage2_full_sft: full SFT (LoRA + 3D encoder + single-token projection), init from Stage1 ``adapter_path``;
    forward supports mixed batches of LMDB (3D slot) and plain text (same data pipeline as Stage2.py).
  - stage3_full_sft: same full SFT stack; loads via ``init_ckpt`` / ``3D_encoder_ckpt`` as used by Property / NiComplex / Vaska;
    forward is single-token 3D path only (no mixed batch).

Stage2 and Stage3 are both full-SFT recipes; they differ in initialization and Stage2's mixed batching.

Ablation variants (same file):
  - MultimodalModelMultiToken + MultiTokenQueryProjector (multi_token)
  - MultimodalModelFreeze3D / MultimodalModelRandom3D (freeze_3d / random_3d)
"""
from __future__ import annotations

import json
import os
from argparse import Namespace
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import utils  # noqa: F401
from utils import (
    ATOM_DIM,
    MAX_SEQ_LENGTH,
    UNIMOL_MAX_SEQ_LEN,
    OBJECT_REF_CHAT_SEP,
    THREE_D_ENCODER_STATE_PT,
    SINGLE_TOKEN_PROJECTION_PT,
    SINGLE_TOKEN_PROJECTION_SAFETENSORS,
    SingleTokenProjectionLayer,
    build_batch_multi,
    build_embeds_single_token,
    build_embeds_text_only,
    extract_single_token_repr,
    strip_single_token_projection_state_dict,
    MultimodalFullTrainer,
    unwrap_hf_model,
    _pad_stack_embeds,
)

RECIPE_STAGE1 = "stage1_frozen_llm"
RECIPE_STAGE2 = "stage2_full_sft"
RECIPE_STAGE3 = "stage3_full_sft"

_LORA_TARGET_MAP = {
    "qv": ["q_proj", "v_proj"],
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def _unimol_args(dictionary) -> Namespace:
    return Namespace(
        encoder_layers=15,
        encoder_embed_dim=512,
        encoder_ffn_embed_dim=2048,
        encoder_attention_heads=64,
        emb_dropout=0.1,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        pooler_dropout=0.1,
        max_seq_len=512,
        activation_fn="gelu",
        pooler_activation_fn="tanh",
        post_ln=False,
        masked_token_loss=-1.0,
        masked_coord_loss=-1.0,
        masked_dist_loss=-1.0,
        x_norm_loss=-1.0,
        delta_pair_repr_norm_loss=-1.0,
        mode="infer",
    )


def _resolve_dict_path(three_d_encoder_dict_path: Optional[str]) -> str:
    if not three_d_encoder_dict_path:
        raise ValueError("three_d_encoder_dict is required and cannot be empty.")
    if not os.path.isfile(three_d_encoder_dict_path):
        raise FileNotFoundError(f"3D encoder dictionary not found: {three_d_encoder_dict_path}")
    return three_d_encoder_dict_path


def _load_3d_encoder_state(three_d_encoder_ckpt: str):
    if os.path.isdir(three_d_encoder_ckpt):
        sf_path = os.path.join(three_d_encoder_ckpt, "model.safetensors")
        if os.path.isfile(sf_path):
            return load_file(sf_path)
        raise FileNotFoundError(f"No model.safetensors found under {three_d_encoder_ckpt}")
    state = torch.load(three_d_encoder_ckpt, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    return state


def _strip_module_prefix(state):
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}


class MultimodalModel(nn.Module):
    """Single implementation for single-token-slot 3D + Qwen; select behavior via ``recipe``."""

    def __init__(
        self,
        model_name: str,
        three_d_encoder_dict_path: str,
        *,
        recipe: str,
        adapter_path: Optional[str] = None,
        three_d_encoder_ckpt: Optional[str] = None,
        init_ckpt: Optional[str] = None,
        train_3d_encoder: Optional[bool] = None,
        train_projection: Optional[bool] = None,
        train_lora: bool = True,
        load_pretrained_projection: bool = True,
        load_pretrained_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_target: str = "qv",
        include_single_token: bool = True,
        single_token_only: bool = True,
        require_3d_encoder: bool = True,
    ):
        super().__init__()
        if recipe not in (RECIPE_STAGE1, RECIPE_STAGE2, RECIPE_STAGE3):
            raise ValueError(f"Unknown recipe={recipe!r}")

        if recipe == RECIPE_STAGE3:
            if not single_token_only:
                raise NotImplementedError(
                    "3DTMC-LLM only implements the single-token-only generative path."
                )
        self.recipe = recipe
        self.single_token_only = True
        self.include_single_token = True
        self.projection_layer = None

        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._lora_target_modules = _LORA_TARGET_MAP.get(lora_target, ["q_proj", "v_proj"])
        self.supports_mixed_batch = recipe == RECIPE_STAGE2
        self._train_3d_encoder = train_3d_encoder
        self._train_projection = train_projection
        self._train_lora = train_lora
        self._load_pretrained_projection = load_pretrained_projection
        self._load_pretrained_lora = load_pretrained_lora
        self._require_3d_encoder = require_3d_encoder

        in_distributed = os.environ.get("LOCAL_RANK") is not None
        device_map = None if in_distributed else "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        from unicore.data import Dictionary

        dict_path = _resolve_dict_path(three_d_encoder_dict_path)
        self.dictionary = Dictionary.load(dict_path)
        self.dictionary.add_symbol("[MASK]", is_special=True)
        self._pad_idx = self.dictionary.pad()
        self._single_token_idx = self.dictionary.bos()
        self._eos_idx = self.dictionary.eos()

        from unimol import UniMolModel

        self.unimol = UniMolModel(_unimol_args(self.dictionary), self.dictionary)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        if recipe == RECIPE_STAGE1:
            if not three_d_encoder_ckpt:
                raise ValueError("stage1_frozen_llm requires three_d_encoder_ckpt")
            state_to_load = _strip_module_prefix(_load_3d_encoder_state(three_d_encoder_ckpt))
            self.unimol.load_state_dict(state_to_load, strict=False)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"[3DTMC-LLM] stage1: Loaded 3D encoder from {three_d_encoder_ckpt}")
            for p in self.unimol.parameters():
                p.requires_grad = True

            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                use_cache=False,
            )
            for p in self.llm.parameters():
                p.requires_grad = False
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print("[3DTMC-LLM] stage1: LLM frozen")

        elif recipe == RECIPE_STAGE2:
            if not adapter_path:
                raise ValueError("stage2_full_sft requires adapter_path (Stage1 single-token-only output directory)")
            three_d_encoder_pt = os.path.join(adapter_path, THREE_D_ENCODER_STATE_PT)
            if not os.path.isfile(three_d_encoder_pt):
                raise FileNotFoundError(
                    f"stage2_full_sft requires {THREE_D_ENCODER_STATE_PT} under adapter_path, missing: {three_d_encoder_pt}"
                )
            state = torch.load(three_d_encoder_pt, map_location="cpu", weights_only=False)
            if "model" in state:
                state = state["model"]
            self.unimol.load_state_dict(state, strict=False)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"[3DTMC-LLM] stage2: Loaded 3D encoder from {three_d_encoder_pt}")
            for p in self.unimol.parameters():
                p.requires_grad = True

            llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                use_cache=False,
            )
            llm = prepare_model_for_kbit_training(llm)
            llm.gradient_checkpointing_enable()
            lora_adapter_path = os.path.join(adapter_path, "adapter_config.json")
            if os.path.isfile(lora_adapter_path):
                self.llm = PeftModel.from_pretrained(llm, adapter_path)
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[3DTMC-LLM] stage2: LoRA from {adapter_path}")
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self._lora_r,
                    lora_alpha=self._lora_alpha,
                    lora_dropout=0.1,
                    target_modules=self._lora_target_modules,
                    bias="none",
                )
                self.llm = get_peft_model(llm, lora_config)
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(
                        f"[3DTMC-LLM] stage2: new LoRA r={self._lora_r}, alpha={self._lora_alpha}, "
                        f"targets={self._lora_target_modules}"
                    )
            self.llm.print_trainable_parameters()

        else:  # RECIPE_STAGE3
            llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                use_cache=False,
            )
            llm = prepare_model_for_kbit_training(llm)
            llm.gradient_checkpointing_enable()

            device = next(llm.parameters()).device
            with torch.no_grad():
                dummy = torch.tensor([[self.tokenizer.eos_token_id]], device=device)
                emb = llm.get_input_embeddings()(dummy)
            hidden_size = emb.shape[-1]
            self.single_token_projection_layer = SingleTokenProjectionLayer(hidden_size).to(device=device, dtype=emb.dtype)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print("[3DTMC-LLM] stage3: single-token -> single-token projection -> LLM")

            if init_ckpt and os.path.isdir(init_ckpt):
                if self._require_3d_encoder:
                    if three_d_encoder_ckpt:
                        three_d_encoder_load_path = three_d_encoder_ckpt
                    else:
                        three_d_encoder_load_path = os.path.join(init_ckpt, THREE_D_ENCODER_STATE_PT)
                    if not (os.path.isfile(three_d_encoder_load_path) or os.path.isdir(three_d_encoder_load_path)):
                        raise FileNotFoundError(
                            f"3D encoder checkpoint missing or invalid: {three_d_encoder_load_path}"
                        )
                    state = _strip_module_prefix(_load_3d_encoder_state(three_d_encoder_load_path))
                    self.unimol.load_state_dict(state, strict=False)
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"[3DTMC-LLM] Loaded 3D encoder from {three_d_encoder_load_path}")
                elif int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print("[3DTMC-LLM] Skipping 3D encoder load (not required for this model)")

                adapter_config = os.path.join(init_ckpt, "adapter_config.json")
                if os.path.isfile(adapter_config) and self._load_pretrained_lora:
                    self.llm = PeftModel.from_pretrained(llm, init_ckpt, is_trainable=True)
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"[3DTMC-LLM] Loaded LoRA from {init_ckpt}")
                else:
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=self._lora_r,
                        lora_alpha=self._lora_alpha,
                        lora_dropout=0.1,
                        target_modules=self._lora_target_modules,
                        bias="none",
                    )
                    self.llm = get_peft_model(llm, lora_config)
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        if os.path.isfile(adapter_config) and not self._load_pretrained_lora:
                            print(
                                f"[3DTMC-LLM] Created new LoRA (r={self._lora_r}); "
                                "skipped Stage2 adapter (load_pretrained_lora=False)"
                            )
                        else:
                            print(f"[3DTMC-LLM] Created new LoRA (r={self._lora_r})")
            else:
                loaded_three_d_encoder = False
                if three_d_encoder_ckpt:
                    if os.path.isdir(three_d_encoder_ckpt):
                        safetensor_path = os.path.join(three_d_encoder_ckpt, "model.safetensors")
                        if os.path.isfile(safetensor_path):
                            state = load_file(safetensor_path, device="cpu")
                            self.unimol.load_state_dict(_strip_module_prefix(state), strict=False)
                            loaded_three_d_encoder = True
                            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                                print(f"[3DTMC-LLM] Loaded 3D encoder from {safetensor_path}")
                    elif os.path.isfile(three_d_encoder_ckpt):
                        if three_d_encoder_ckpt.endswith(".safetensors"):
                            state = load_file(three_d_encoder_ckpt, device="cpu")
                        else:
                            state = torch.load(three_d_encoder_ckpt, map_location="cpu", weights_only=False)
                            if "model" in state:
                                state = state["model"]
                        self.unimol.load_state_dict(_strip_module_prefix(state), strict=False)
                        loaded_three_d_encoder = True
                        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                            print(f"[3DTMC-LLM] Loaded 3D encoder from {three_d_encoder_ckpt}")
                if not loaded_three_d_encoder:
                    raise FileNotFoundError(
                        "3D encoder checkpoint not found or invalid. "
                        "Please provide a valid --3D_encoder_ckpt (file/dir) or ensure "
                        "Stage2_ckpt contains 3D_encoder.pt."
                    )

                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self._lora_r,
                    lora_alpha=self._lora_alpha,
                    lora_dropout=0.1,
                    target_modules=self._lora_target_modules,
                    bias="none",
                )
                self.llm = get_peft_model(llm, lora_config)
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[3DTMC-LLM] Created new LoRA (r={self._lora_r})")

            self.llm.print_trainable_parameters()

            if init_ckpt and os.path.isdir(init_ckpt):
                three_d_encoder_pt_in_ckpt = (
                    three_d_encoder_ckpt
                    if (three_d_encoder_ckpt and (os.path.isfile(three_d_encoder_ckpt) or os.path.isdir(three_d_encoder_ckpt)))
                    else os.path.join(init_ckpt, THREE_D_ENCODER_STATE_PT)
                )
                if os.path.isfile(three_d_encoder_pt_in_ckpt) or os.path.isdir(three_d_encoder_pt_in_ckpt):
                    three_d_encoder_source = three_d_encoder_pt_in_ckpt
                else:
                    three_d_encoder_source = "random_init"
                lora_source = (
                    init_ckpt
                    if (
                        self._load_pretrained_lora
                        and os.path.isfile(os.path.join(init_ckpt, "adapter_config.json"))
                    )
                    else "new"
                )
            else:
                three_d_encoder_source = three_d_encoder_ckpt if three_d_encoder_ckpt else "random_init"
                lora_source = "new"

            if self._train_3d_encoder is None:
                self._train_3d_encoder = not (init_ckpt and os.path.isdir(init_ckpt))
            if self._train_projection is None:
                self._train_projection = not (init_ckpt and os.path.isdir(init_ckpt))

            for p in self.unimol.parameters():
                p.requires_grad = self._train_3d_encoder
            if not self._train_lora:
                for n, p in self.llm.named_parameters():
                    if "lora" in n.lower():
                        p.requires_grad = False

            projection_source = "new"
            if self._load_pretrained_projection and init_ckpt and os.path.isdir(init_ckpt):
                st_path = os.path.join(init_ckpt, SINGLE_TOKEN_PROJECTION_SAFETENSORS)
                pt_path = os.path.join(init_ckpt, SINGLE_TOKEN_PROJECTION_PT)

                if os.path.isfile(st_path):
                    raw = load_file(st_path, device="cpu")
                    sd = strip_single_token_projection_state_dict(dict(raw))
                    if sd:
                        self.single_token_projection_layer.load_state_dict(sd, strict=False)
                    projection_source = st_path
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"[3DTMC-LLM] Loaded single-token projection from {st_path}")
                elif os.path.isfile(pt_path):
                    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
                    sd = strip_single_token_projection_state_dict(raw if isinstance(raw, dict) else {})
                    if sd:
                        self.single_token_projection_layer.load_state_dict(sd, strict=False)
                    projection_source = pt_path
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"[3DTMC-LLM] Loaded single-token projection from {pt_path}")
                elif int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print("[3DTMC-LLM] single-token projection trained from scratch")
            elif int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print("[3DTMC-LLM] single-token projection forced from-scratch init")

            train_projection = self._train_projection if self._train_projection is not None else True
            for p in self.single_token_projection_layer.parameters():
                p.requires_grad = train_projection

            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                self._print_module_config_table(three_d_encoder_source, lora_source, projection_source)

        if recipe != RECIPE_STAGE3:
            device = next(self.llm.parameters()).device
            with torch.no_grad():
                dummy = torch.tensor([[self.tokenizer.eos_token_id]], device=device)
                emb = self.llm.get_input_embeddings()(dummy)
            hidden_size = emb.shape[-1]
            self.single_token_projection_layer = SingleTokenProjectionLayer(hidden_size).to(device=device, dtype=emb.dtype)
            if recipe == RECIPE_STAGE2:
                st_path = os.path.join(adapter_path, SINGLE_TOKEN_PROJECTION_SAFETENSORS)
                if os.path.isfile(st_path):
                    raw = load_file(st_path)
                    sd = strip_single_token_projection_state_dict(dict(raw))
                    if sd:
                        self.single_token_projection_layer.load_state_dict(sd, strict=False)
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"[3DTMC-LLM] Loaded single-token projection from {st_path}")
                else:
                    pt_path = os.path.join(adapter_path, SINGLE_TOKEN_PROJECTION_PT)
                    if os.path.isfile(pt_path):
                        raw = torch.load(pt_path, map_location="cpu", weights_only=False)
                        sd = strip_single_token_projection_state_dict(raw if isinstance(raw, dict) else {})
                        if sd:
                            self.single_token_projection_layer.load_state_dict(sd, strict=False)
                        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                            print(f"[3DTMC-LLM] Loaded single-token projection from {pt_path}")
                for p in self.single_token_projection_layer.parameters():
                    p.requires_grad = True
            else:
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[3DTMC-LLM] stage1: single-token projection ({ATOM_DIM}->{hidden_size})")

        self._start_3d_id = self.tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
        self._end_3d_id = self.tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
        self.unimol = self.unimol.to(next(self.llm.parameters()).device)

    def _print_module_config_table(self, three_d_encoder_source, lora_source, projection_source="new"):
        def count_params(module):
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters())

        unimol_params = count_params(self.unimol)
        projection_params = count_params(self.single_token_projection_layer)
        lora_params = sum(p.numel() for n, p in self.llm.named_parameters() if "lora" in n.lower())

        def fmt_params(n):
            if n == 0:
                return "-"
            if n >= 1e9:
                return f"{n/1e9:.2f}B"
            if n >= 1e6:
                return f"{n/1e6:.2f}M"
            if n >= 1e3:
                return f"{n/1e3:.2f}K"
            return str(n)

        train_projection = self._train_projection if self._train_projection is not None else True
        print("\n" + "=" * 90)
        print("                         Module config (single-token-only)")
        print("=" * 90)
        print(f"{'Module':<20} {'Train':<12} {'#Params':<15} {'Source'}")
        print("-" * 90)
        print(f"{'3D encoder':<20} {'train' if self._train_3d_encoder else 'frozen':<12} {fmt_params(unimol_params):<15} {three_d_encoder_source}")
        print(f"{'single-token proj.':<20} {'train' if train_projection else 'frozen':<12} {fmt_params(projection_params):<15} {projection_source}")
        print(f"{'LoRA':<20} {'train' if self._train_lora else 'frozen':<12} {fmt_params(lora_params):<15} {lora_source}")
        print("-" * 90)
        total_trainable = 0
        if self._train_3d_encoder:
            total_trainable += unimol_params
        if train_projection:
            total_trainable += projection_params
        if self._train_lora:
            total_trainable += lora_params
        print(f"{'Total trainable':<20} {fmt_params(total_trainable)}")
        print("=" * 90 + "\n")

    def _forward_3d_batch(
        self,
        device,
        before_3d_ids,
        before_3d_mask,
        after_3d_ids,
        after_3d_mask,
        response_ids,
        response_mask,
        list_atoms,
        list_coordinates,
    ):
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
        single_token_repr = extract_single_token_repr(encoder_rep)
        return build_embeds_single_token(
            self.llm,
            device,
            before_3d_ids,
            before_3d_mask,
            after_3d_ids,
            after_3d_mask,
            response_ids,
            response_mask,
            single_token_repr,
            self.single_token_projection_layer,
            self._start_3d_id,
            self._end_3d_id,
        )

    def _get_3d_encoder_single_token(self, list_atoms, list_coordinates, device):
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
        return extract_single_token_repr(encoder_rep)

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
        sample_types: Optional[List[str]] = None,
        **kwargs,
    ):
        device = next(self.llm.parameters()).device

        if self.recipe == RECIPE_STAGE3:
            single_token_repr = self._get_3d_encoder_single_token(list_atoms, list_coordinates, device)
            inputs_embeds, attention_mask, labels = build_embeds_single_token(
                self.llm,
                device,
                before_3d_ids,
                before_3d_mask,
                after_3d_ids,
                after_3d_mask,
                response_ids,
                response_mask,
                single_token_repr,
                self.single_token_projection_layer,
                self._start_3d_id,
                self._end_3d_id,
            )
            return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

        if self.recipe == RECIPE_STAGE1:
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
            single_token_repr = extract_single_token_repr(encoder_rep)
            inputs_embeds, attention_mask, labels = build_embeds_single_token(
                self.llm,
                device,
                before_3d_ids,
                before_3d_mask,
                after_3d_ids,
                after_3d_mask,
                response_ids,
                response_mask,
                single_token_repr,
                self.single_token_projection_layer,
                self._start_3d_id,
                self._end_3d_id,
            )
            if inputs_embeds.shape[1] > MAX_SEQ_LENGTH:
                inputs_embeds = inputs_embeds[:, :MAX_SEQ_LENGTH, :]
                attention_mask = attention_mask[:, :MAX_SEQ_LENGTH]
                labels = labels[:, :MAX_SEQ_LENGTH]
            return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

        # stage2_full_sft: mixed LMDB / text (Stage2 data pipeline)
        B = before_3d_ids.shape[0]
        sample_types = sample_types if sample_types is not None else ["lmdb"] * B
        has_3d = list_atoms and any(len(a) > 0 for a in list_atoms)
        all_text = all(t == "text" for t in sample_types)
        all_lmdb = all(t == "lmdb" for t in sample_types)

        if not has_3d or all_text:
            inputs_embeds, attention_mask, labels = build_embeds_text_only(
                self.llm, device, before_3d_ids, before_3d_mask, response_ids, response_mask
            )
        elif all_lmdb and has_3d:
            inputs_embeds, attention_mask, labels = self._forward_3d_batch(
                device,
                before_3d_ids,
                before_3d_mask,
                after_3d_ids,
                after_3d_mask,
                response_ids,
                response_mask,
                list_atoms,
                list_coordinates,
            )
        else:
            text_idx = [i for i, t in enumerate(sample_types) if t == "text"]
            lmdb_idx = [i for i, t in enumerate(sample_types) if t == "lmdb"]
            emb_list = []
            if text_idx:
                b3 = before_3d_ids[text_idx]
                b3m = before_3d_mask[text_idx]
                rids = response_ids[text_idx]
                rmask = response_mask[text_idx]
                inp, attn, lbl = build_embeds_text_only(self.llm, device, b3, b3m, rids, rmask)
                emb_list.append((text_idx, inp, attn, lbl))
            if lmdb_idx:
                b3 = before_3d_ids[lmdb_idx]
                b3m = before_3d_mask[lmdb_idx]
                a3 = after_3d_ids[lmdb_idx]
                a3m = after_3d_mask[lmdb_idx]
                rids = response_ids[lmdb_idx]
                rmask = response_mask[lmdb_idx]
                atoms_list = [list_atoms[i] for i in lmdb_idx]
                coords_list = [list_coordinates[i] for i in lmdb_idx]
                inp, attn, lbl = self._forward_3d_batch(
                    device, b3, b3m, a3, a3m, rids, rmask, atoms_list, coords_list
                )
                emb_list.append((lmdb_idx, inp, attn, lbl))
            max_L = max(e.shape[1] for _, e, _, _ in emb_list)
            H = emb_list[0][1].shape[2]
            dtype_emb = emb_list[0][1].dtype
            out_emb = torch.zeros(B, max_L, H, device=device, dtype=dtype_emb)
            out_mask = torch.zeros(B, max_L, dtype=torch.long, device=device)
            out_labels = torch.full((B, max_L), -100, dtype=torch.long, device=device)
            for indices, e, m, lbl in emb_list:
                L = e.shape[1]
                for ii, i in enumerate(indices):
                    out_emb[i, :L] = e[ii]
                    out_mask[i, :L] = m[ii]
                    out_labels[i, :L] = lbl[ii]
            inputs_embeds = out_emb
            attention_mask = out_mask
            labels = out_labels

        if inputs_embeds.shape[1] > MAX_SEQ_LENGTH:
            inputs_embeds = inputs_embeds[:, :MAX_SEQ_LENGTH, :]
            attention_mask = attention_mask[:, :MAX_SEQ_LENGTH]
            labels = labels[:, :MAX_SEQ_LENGTH]

        valid_label_count = (labels != -100).sum().item()
        if valid_label_count == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"[WARNING] All labels are -100! seq_len={labels.shape[1]}, sample_types={sample_types}, "
                f"response_mask_sum={response_mask.sum(dim=1).tolist()}, "
                f"before_3d_mask_sum={before_3d_mask.sum(dim=1).tolist()}"
            )

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)


def generate_with_single_token_structure(
    model: MultimodalModel,
    tokenizer,
    atoms,
    coords,
    user_content: str,
    *,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
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
    if sep not in prefix_str:
        before_3d_str = prefix_str
        after_3d_str = ""
    else:
        before_3d_str, rest = prefix_str.split(sep, 1)
        after_3d_str = sep + rest

    with torch.inference_mode():
        before_3d_ids = tokenizer(before_3d_str, return_tensors="pt").input_ids.to(device)
        after_3d_ids = tokenizer(after_3d_str, return_tensors="pt").input_ids.to(device) if after_3d_str else None
        before_3d_embeds = embed_layer(before_3d_ids)
        if after_3d_ids is not None and after_3d_ids.shape[1] > 0:
            after_3d_embeds = embed_layer(after_3d_ids)
        else:
            after_3d_embeds = torch.empty(
                1, 0, before_3d_embeds.shape[2], device=device, dtype=before_3d_embeds.dtype
            )
        start_ids = torch.tensor([[model._start_3d_id]], device=device)
        end_ids = torch.tensor([[model._end_3d_id]], device=device)
        start_emb = embed_layer(start_ids)
        end_emb = embed_layer(end_ids)

    model_dtype = before_3d_embeds.dtype
    start_emb = start_emb.to(model_dtype)
    end_emb = end_emb.to(model_dtype)
    mol_embeds = mol_embeds.to(model_dtype)
    three_d_block = torch.cat([start_emb, mol_embeds, end_emb], dim=1)
    fused_embeddings = torch.cat([before_3d_embeds, three_d_block, after_3d_embeds], dim=1)
    fused_attention_mask = torch.ones((1, fused_embeddings.shape[1]), dtype=torch.long, device=device)

    eos_id = tokenizer.eos_token_id
    prompt_len = int(fused_attention_mask[0].sum().item())
    model.llm.eval()

    gen_kwargs = dict(
        inputs_embeds=fused_embeddings,
        attention_mask=fused_attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = max(temperature, 1e-5)
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        out_ids = model.llm.generate(**gen_kwargs)
    # With inputs_embeds, HF generate often returns *only* new token ids (len << prompt_len).
    if out_ids.shape[1] > prompt_len:
        gen_ids = out_ids[0, prompt_len:]
    else:
        gen_ids = out_ids[0]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()



# ---------- Multi-token query projection (ablation) ----------

NUM_STRUCTURE_QUERIES = 8
MULTI_TOKEN_PROJECTION_PT = "multi_token_projection.pt"
LEGACY_MULTI_TOKEN_PROJECTION_PT = "k8_projection.pt"


def resolve_multi_token_projection_path(ckpt_dir: str) -> str | None:
    """Resolve projection weights (new name first, then legacy k8_projection.pt)."""
    for name in (MULTI_TOKEN_PROJECTION_PT, LEGACY_MULTI_TOKEN_PROJECTION_PT):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            return path
    return None


def extract_atom_repr_padded(
    encoder_rep: torch.Tensor,
    list_atoms: Sequence[Sequence],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Atom positions ``encoder_rep[:, 1:1+n_atoms]`` padded to batch max length."""
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


class MultiTokenQueryProjector(nn.Module):
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
            raise RuntimeError("MultiTokenQueryProjector: no atom tokens in batch (empty structures).")
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


def build_embeds_multi_token(
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
    multi_token_projection_layer: MultiTokenQueryProjector,
    start_3d_id,
    end_3d_id,
):
    """Insert K projected structure vectors between object_ref start/end token embeddings."""
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

    proj_dtype = next(multi_token_projection_layer.parameters()).dtype
    atom_repr = atom_repr.to(device=device, dtype=proj_dtype)
    structure_proj = multi_token_projection_layer(atom_repr, atom_pad_mask.to(device)).to(dtype=dtype_emb)

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


# ---------- Multi-token multimodal model ----------


class MultimodalModelMultiToken(MultimodalModel):
    """Uni-Mol atom hiddens -> learnable query tokens (cross-attn) -> LLM embeddings."""

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
        if self.recipe not in (RECIPE_STAGE1, RECIPE_STAGE3):
            raise ValueError("MultimodalModelMultiToken supports recipe=stage1_frozen_llm or stage3_full_sft")
        hidden_size = self.single_token_projection_layer.projection.out_features
        device = next(self.llm.parameters()).device
        dtype = next(self.single_token_projection_layer.parameters()).dtype
        self.multi_token_projection_layer = MultiTokenQueryProjector(
            ATOM_DIM, hidden_size, num_queries=NUM_STRUCTURE_QUERIES
        ).to(device=device, dtype=dtype)
        self.single_token_projection_layer = None
        train_projection = self._train_projection if self._train_projection is not None else True
        for p in self.multi_token_projection_layer.parameters():
            p.requires_grad = train_projection
        if init_ckpt and os.path.isdir(init_ckpt):
            proj_path = resolve_multi_token_projection_path(init_ckpt)
            if proj_path:
                state = torch.load(proj_path, map_location="cpu", weights_only=True)
                self.multi_token_projection_layer.load_state_dict(state, strict=True)
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[3DTMC-LLM-multi-token] Loaded projection from {proj_path}")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if self.recipe == RECIPE_STAGE1:
                print(
                    f"[3DTMC-LLM-multi-token] stage1: {NUM_STRUCTURE_QUERIES} query tokens; "
                    f"frozen LLM; train 3D encoder + multi-token projection ({ATOM_DIM}->{hidden_size})"
                )
            else:
                lora_note = "Stage2 LoRA" if self._load_pretrained_lora else "new LoRA on base Qwen (from scratch)"
                print(
                    f"[3DTMC-LLM-multi-token] Uni-Mol atom tokens -> {NUM_STRUCTURE_QUERIES} learnable queries "
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
        inputs_embeds, attention_mask, labels = build_embeds_multi_token(
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
            self.multi_token_projection_layer,
            self._start_3d_id,
            self._end_3d_id,
        )
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)


class MultimodalMultiTokenTrainer(MultimodalFullTrainer):
    """Saves multi_token_projection.pt instead of single_token_projection.pt."""

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)
        model = unwrap_hf_model(self.model)
        model.llm.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
        if getattr(model, "multi_token_projection_layer", None) is not None:
            state = {k: v.cpu() for k, v in model.multi_token_projection_layer.state_dict().items()}
            torch.save(state, os.path.join(output_dir, MULTI_TOKEN_PROJECTION_PT))
        torch.save({"model": model.unimol.state_dict()}, os.path.join(output_dir, THREE_D_ENCODER_STATE_PT))
        with open(os.path.join(output_dir, "multimodal_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "atom_dim": ATOM_DIM,
                    "structure_tokens": "multi_token_query",
                    "num_structure_queries": NUM_STRUCTURE_QUERIES,
                    "include_single_token": False,
                    "single_token_only": False,
                    "load_pretrained_lora": getattr(model, "_load_pretrained_lora", False),
                },
                f,
                indent=2,
            )


# ---------- Freeze-3D / random-3D ablations ----------


def make_random_structure_embeddings(
    sample_indices: List[int],
    hidden_size: int,
    base_seed: int,
    device,
    dtype,
) -> torch.Tensor:
    """Deterministic per-sample random LLM-space embeddings [B, 1, hidden_size]."""
    vectors = []
    for idx in sample_indices:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(base_seed + int(idx))
        vectors.append(torch.randn(1, hidden_size, generator=gen, dtype=torch.float32))
    stacked = torch.cat(vectors, dim=0).to(device=device, dtype=dtype)
    return stacked.unsqueeze(1)


def build_embeds_precomputed_structure(
    llm,
    device,
    before_3d_ids,
    before_3d_mask,
    after_3d_ids,
    after_3d_mask,
    response_ids,
    response_mask,
    structure_token_emb,
    start_3d_id,
    end_3d_id,
):
    """Insert precomputed structure embeddings at the object_ref single-token slot."""
    embed_fn = llm.get_input_embeddings()
    batch_size = before_3d_ids.shape[0]
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
    structure_token_emb = structure_token_emb.to(device=device, dtype=dtype_emb)

    l1 = before_3d_mask.sum(dim=1)
    l2 = after_3d_mask.sum(dim=1)
    lr = response_mask.sum(dim=1)
    emb_list, mask_list, label_list = [], [], []
    for i in range(batch_size):
        l1_i, l2_i, lr_i = l1[i].item(), l2[i].item(), lr[i].item()
        if lr_i == 0:
            raise RuntimeError("response effective length is 0 (Lr=0).")
        single_token_emb = structure_token_emb[i : i + 1]
        fused = torch.cat(
            [
                before_3d_embeds[i : i + 1, :l1_i],
                start_emb.expand(1, -1, -1),
                single_token_emb,
                end_emb.expand(1, -1, -1),
                after_3d_embeds[i : i + 1, :l2_i],
            ],
            dim=1,
        )
        fused_mask = torch.ones(1, fused.shape[1], dtype=torch.long, device=device)
        fused_labels = torch.full((1, fused.shape[1]), -100, dtype=torch.long, device=device)
        all_emb = torch.cat([fused, response_embeds[i : i + 1, :lr_i]], dim=1)
        all_mask = torch.cat([fused_mask, response_mask[i : i + 1, :lr_i]], dim=1)
        resp_labels = response_ids[i : i + 1, :lr_i].clone()
        resp_labels[response_mask[i : i + 1, :lr_i] == 0] = -100
        all_labels = torch.cat([fused_labels, resp_labels], dim=1)
        emb_list.append(all_emb)
        mask_list.append(all_mask)
        label_list.append(all_labels)


    return _pad_stack_embeds(emb_list, mask_list, label_list, device=device, dtype_emb=dtype_emb)


class MultimodalModelFreeze3D(MultimodalModel):
    """Stage2-style mixed batch; frozen 3D encoder + projection; train LoRA only."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("recipe", RECIPE_STAGE3)
        super().__init__(*args, **kwargs)

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
        sample_types: Optional[List[str]] = None,
        sample_indices=None,
        **kwargs,
    ):
        device = next(self.llm.parameters()).device
        batch_size = before_3d_ids.shape[0]
        sample_types = sample_types if sample_types is not None else ["lmdb"] * batch_size
        has_3d = list_atoms and any(len(a) > 0 for a in list_atoms)
        all_text = all(t == "text" for t in sample_types)
        all_lmdb = all(t == "lmdb" for t in sample_types)

        if not has_3d or all_text:
            inputs_embeds, attention_mask, labels = build_embeds_text_only(
                self.llm, device, before_3d_ids, before_3d_mask, response_ids, response_mask
            )
        elif all_lmdb and has_3d:
            inputs_embeds, attention_mask, labels = self._forward_3d_batch(
                device,
                before_3d_ids,
                before_3d_mask,
                after_3d_ids,
                after_3d_mask,
                response_ids,
                response_mask,
                list_atoms,
                list_coordinates,
            )
        else:
            text_idx = [i for i, t in enumerate(sample_types) if t == "text"]
            lmdb_idx = [i for i, t in enumerate(sample_types) if t == "lmdb"]
            emb_list = []
            if text_idx:
                b3 = before_3d_ids[text_idx]
                b3m = before_3d_mask[text_idx]
                rids = response_ids[text_idx]
                rmask = response_mask[text_idx]
                inp, attn, lbl = build_embeds_text_only(self.llm, device, b3, b3m, rids, rmask)
                emb_list.append((text_idx, inp, attn, lbl))
            if lmdb_idx:
                b3 = before_3d_ids[lmdb_idx]
                b3m = before_3d_mask[lmdb_idx]
                a3 = after_3d_ids[lmdb_idx]
                a3m = after_3d_mask[lmdb_idx]
                rids = response_ids[lmdb_idx]
                rmask = response_mask[lmdb_idx]
                atoms_list = [list_atoms[i] for i in lmdb_idx]
                coords_list = [list_coordinates[i] for i in lmdb_idx]
                inp, attn, lbl = self._forward_3d_batch(
                    device, b3, b3m, a3, a3m, rids, rmask, atoms_list, coords_list
                )
                emb_list.append((lmdb_idx, inp, attn, lbl))
            max_l = max(e.shape[1] for _, e, _, _ in emb_list)
            hidden = emb_list[0][1].shape[2]
            dtype_emb = emb_list[0][1].dtype
            out_emb = torch.zeros(batch_size, max_l, hidden, device=device, dtype=dtype_emb)
            out_mask = torch.zeros(batch_size, max_l, dtype=torch.long, device=device)
            out_labels = torch.full((batch_size, max_l), -100, dtype=torch.long, device=device)
            for indices, e, m, lbl in emb_list:
                length = e.shape[1]
                for ii, i in enumerate(indices):
                    out_emb[i, :length] = e[ii]
                    out_mask[i, :length] = m[ii]
                    out_labels[i, :length] = lbl[ii]
            inputs_embeds = out_emb
            attention_mask = out_mask
            labels = out_labels

        if inputs_embeds.shape[1] > MAX_SEQ_LENGTH:
            inputs_embeds = inputs_embeds[:, :MAX_SEQ_LENGTH, :]
            attention_mask = attention_mask[:, :MAX_SEQ_LENGTH]
            labels = labels[:, :MAX_SEQ_LENGTH]
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)


class MultimodalModelRandom3D(MultimodalModel):
    """Stage2-style mixed batch; random structure-slot embedding; train LoRA only."""

    def __init__(self, *args, random_3d_seed: int = 42, **kwargs):
        kwargs.setdefault("recipe", RECIPE_STAGE3)
        kwargs.setdefault("require_3d_encoder", False)
        super().__init__(*args, **kwargs)
        self.random_3d_seed = int(random_3d_seed)

    def _forward_random_3d_batch(
        self,
        device,
        before_3d_ids,
        before_3d_mask,
        after_3d_ids,
        after_3d_mask,
        response_ids,
        response_mask,
        sample_indices,
    ):
        embed_layer = self.llm.get_input_embeddings()
        hidden_size = embed_layer.embedding_dim
        dtype_emb = embed_layer.weight.dtype
        random_emb = make_random_structure_embeddings(
            sample_indices,
            hidden_size,
            self.random_3d_seed,
            device,
            dtype_emb,
        )
        return build_embeds_precomputed_structure(
            self.llm,
            device,
            before_3d_ids,
            before_3d_mask,
            after_3d_ids,
            after_3d_mask,
            response_ids,
            response_mask,
            random_emb,
            self._start_3d_id,
            self._end_3d_id,
        )

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
        sample_types: Optional[List[str]] = None,
        sample_indices=None,
        **kwargs,
    ):
        device = next(self.llm.parameters()).device
        batch_size = before_3d_ids.shape[0]
        sample_types = sample_types if sample_types is not None else ["lmdb"] * batch_size
        sample_indices = sample_indices if sample_indices is not None else [-1] * batch_size
        has_3d = list_atoms and any(len(a) > 0 for a in list_atoms)
        all_text = all(t == "text" for t in sample_types)
        all_lmdb = all(t == "lmdb" for t in sample_types)

        if not has_3d or all_text:
            inputs_embeds, attention_mask, labels = build_embeds_text_only(
                self.llm, device, before_3d_ids, before_3d_mask, response_ids, response_mask
            )
        elif all_lmdb and has_3d:
            inputs_embeds, attention_mask, labels = self._forward_random_3d_batch(
                device,
                before_3d_ids,
                before_3d_mask,
                after_3d_ids,
                after_3d_mask,
                response_ids,
                response_mask,
                sample_indices,
            )
        else:
            text_idx = [i for i, t in enumerate(sample_types) if t == "text"]
            lmdb_idx = [i for i, t in enumerate(sample_types) if t == "lmdb"]
            emb_list = []
            if text_idx:
                b3 = before_3d_ids[text_idx]
                b3m = before_3d_mask[text_idx]
                rids = response_ids[text_idx]
                rmask = response_mask[text_idx]
                inp, attn, lbl = build_embeds_text_only(self.llm, device, b3, b3m, rids, rmask)
                emb_list.append((text_idx, inp, attn, lbl))
            if lmdb_idx:
                b3 = before_3d_ids[lmdb_idx]
                b3m = before_3d_mask[lmdb_idx]
                a3 = after_3d_ids[lmdb_idx]
                a3m = after_3d_mask[lmdb_idx]
                rids = response_ids[lmdb_idx]
                rmask = response_mask[lmdb_idx]
                lmdb_sample_indices = [sample_indices[i] for i in lmdb_idx]
                inp, attn, lbl = self._forward_random_3d_batch(
                    device, b3, b3m, a3, a3m, rids, rmask, lmdb_sample_indices
                )
                emb_list.append((lmdb_idx, inp, attn, lbl))
            max_l = max(e.shape[1] for _, e, _, _ in emb_list)
            hidden = emb_list[0][1].shape[2]
            dtype_emb = emb_list[0][1].dtype
            out_emb = torch.zeros(batch_size, max_l, hidden, device=device, dtype=dtype_emb)
            out_mask = torch.zeros(batch_size, max_l, dtype=torch.long, device=device)
            out_labels = torch.full((batch_size, max_l), -100, dtype=torch.long, device=device)
            for indices, e, m, lbl in emb_list:
                length = e.shape[1]
                for ii, i in enumerate(indices):
                    out_emb[i, :length] = e[ii]
                    out_mask[i, :length] = m[ii]
                    out_labels[i, :length] = lbl[ii]
            inputs_embeds = out_emb
            attention_mask = out_mask
            labels = out_labels

        if inputs_embeds.shape[1] > MAX_SEQ_LENGTH:
            inputs_embeds = inputs_embeds[:, :MAX_SEQ_LENGTH, :]
            attention_mask = attention_mask[:, :MAX_SEQ_LENGTH]
            labels = labels[:, :MAX_SEQ_LENGTH]
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)


def generate_batch_with_random_structure(
    model: "MultimodalModelRandom3D",
    tokenizer,
    batch_user_contents: List[str],
    batch_sample_indices: List[int],
    *,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Batched greedy/sample decode with deterministic random structure-slot embeddings."""
    if not batch_user_contents:
        return []

    device = next(model.llm.parameters()).device
    embed_layer = model.llm.get_input_embeddings()
    hidden_size = embed_layer.embedding_dim
    dtype_emb = embed_layer.weight.dtype
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    random_emb = make_random_structure_embeddings(
        batch_sample_indices,
        hidden_size,
        model.random_3d_seed,
        device,
        dtype_emb,
    )


    sep = OBJECT_REF_CHAT_SEP
    split_parts = []
    for uc in batch_user_contents:
        prefix_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": uc}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if sep not in prefix_str:
            split_parts.append((prefix_str, ""))
        else:
            before_3d_str, rest = prefix_str.split(sep, 1)
            split_parts.append((before_3d_str, sep + rest))
    before_parts, after_parts = zip(*split_parts)

    before_ids_list = [tokenizer(b, return_tensors="pt").input_ids for b in before_parts]
    after_ids_list = [
        tokenizer(a, return_tensors="pt").input_ids if a else torch.zeros(1, 0, dtype=torch.long)
        for a in after_parts
    ]

    def _pad_token_ids(ids_list, pad_id, device):
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

    before_ids, before_mask = _pad_token_ids(before_ids_list, pad_id, device)
    after_ids, after_mask = _pad_token_ids(after_ids_list, pad_id, device)
    before_embeds = embed_layer(before_ids)
    after_embeds = embed_layer(after_ids)

    start_ids = torch.full((len(batch_user_contents), 1), model._start_3d_id, dtype=torch.long, device=device)
    end_ids = torch.full((len(batch_user_contents), 1), model._end_3d_id, dtype=torch.long, device=device)
    start_emb = embed_layer(start_ids)
    end_emb = embed_layer(end_ids)

    model_dtype = before_embeds.dtype
    start_emb = start_emb.to(model_dtype)
    end_emb = end_emb.to(model_dtype)
    mol_embeds = random_emb.to(model_dtype)
    three_d_block = torch.cat([start_emb, mol_embeds, end_emb], dim=1)
    three_d_mask = torch.ones((len(batch_user_contents), three_d_block.shape[1]), dtype=torch.long, device=device)

    fused_embeddings = torch.cat([before_embeds, three_d_block, after_embeds], dim=1)
    fused_attention_mask = torch.cat([before_mask, three_d_mask, after_mask], dim=1)

    eos_id = tokenizer.eos_token_id
    prompt_lens = fused_attention_mask.sum(dim=1).tolist()
    model.llm.eval()

    gen_kwargs = dict(
        inputs_embeds=fused_embeddings,
        attention_mask=fused_attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = max(temperature, 1e-5)
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        out_ids = model.llm.generate(**gen_kwargs)

    texts = []
    for row, prompt_len in zip(out_ids, prompt_lens):
        gen_ids = row[prompt_len:] if row.shape[0] > prompt_len else row
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return texts


def generate_with_random_structure(
    model: MultimodalModelRandom3D,
    tokenizer,
    user_content: str,
    *,
    sample_idx: int = 0,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Greedy/sample decode with a deterministic random structure-slot embedding."""
    device = next(model.llm.parameters()).device
    embed_layer = model.llm.get_input_embeddings()
    hidden_size = embed_layer.embedding_dim
    dtype_emb = embed_layer.weight.dtype
    random_emb = make_random_structure_embeddings(
        [sample_idx],
        hidden_size,
        model.random_3d_seed,
        device,
        dtype_emb,
    )

    prefix_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    sep = OBJECT_REF_CHAT_SEP
    if sep not in prefix_str:
        before_3d_str = prefix_str
        after_3d_str = ""
    else:
        before_3d_str, rest = prefix_str.split(sep, 1)
        after_3d_str = sep + rest

    with torch.inference_mode():
        before_3d_ids = tokenizer(before_3d_str, return_tensors="pt").input_ids.to(device)
        after_3d_ids = tokenizer(after_3d_str, return_tensors="pt").input_ids.to(device) if after_3d_str else None
        before_3d_embeds = embed_layer(before_3d_ids)
        if after_3d_ids is not None and after_3d_ids.shape[1] > 0:
            after_3d_embeds = embed_layer(after_3d_ids)
        else:
            after_3d_embeds = torch.empty(
                1, 0, before_3d_embeds.shape[2], device=device, dtype=before_3d_embeds.dtype
            )
        start_ids = torch.tensor([[model._start_3d_id]], device=device)
        end_ids = torch.tensor([[model._end_3d_id]], device=device)
        start_emb = embed_layer(start_ids)
        end_emb = embed_layer(end_ids)

    model_dtype = before_3d_embeds.dtype
    start_emb = start_emb.to(model_dtype)
    end_emb = end_emb.to(model_dtype)
    mol_embeds = random_emb.to(model_dtype)
    three_d_block = torch.cat([start_emb, mol_embeds, end_emb], dim=1)
    fused_embeddings = torch.cat([before_3d_embeds, three_d_block, after_3d_embeds], dim=1)
    fused_attention_mask = torch.ones((1, fused_embeddings.shape[1]), dtype=torch.long, device=device)

    eos_id = tokenizer.eos_token_id
    model.llm.eval()
    gen_kwargs = dict(
        inputs_embeds=fused_embeddings,
        attention_mask=fused_attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = max(temperature, 1e-5)
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        out = model.llm.generate(**gen_kwargs)
    if out.shape[1] > 0:
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return ""

__all__ = [
    "MultimodalModel",
    "MultimodalModelMultiToken",
    "MultimodalModelFreeze3D",
    "MultimodalModelRandom3D",
    "MultimodalMultiTokenTrainer",
    "MultiTokenQueryProjector",
    "NUM_STRUCTURE_QUERIES",
    "MULTI_TOKEN_PROJECTION_PT",
    "RECIPE_STAGE1",
    "RECIPE_STAGE2",
    "RECIPE_STAGE3",
    "build_embeds_multi_token",
    "extract_atom_repr_padded",
    "resolve_multi_token_projection_path",
    "generate_with_single_token_structure",
    "generate_with_random_structure",
    "generate_batch_with_random_structure",
    "make_random_structure_embeddings",
]
