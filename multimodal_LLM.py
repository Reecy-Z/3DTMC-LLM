"""
Unified 3D encoder single-token + Qwen3 (4-bit) stack for Stage1 / Stage2 / downstream full SFT.

Recipes (string values):
  - stage1_frozen_llm: frozen LLM; train 3D encoder + single-token projection only.
  - stage2_full_sft: full SFT (LoRA + 3D encoder + single-token projection), init from Stage1 ``adapter_path``;
    forward supports mixed batches of LMDB (3D slot) and plain text (same data pipeline as Stage2.py).
  - stage3_full_sft: same full SFT stack; loads via ``init_ckpt`` / ``3D_encoder_ckpt`` as used by Property / NiComplex / Vaska;
    forward is single-token 3D path only (no mixed batch).

Stage2 and Stage3 are both full-SFT recipes; they differ in initialization and Stage2's mixed batching.
"""
from __future__ import annotations

import os
from argparse import Namespace
from typing import List, Optional

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
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_target: str = "qv",
        include_single_token: bool = True,
        single_token_only: bool = True,
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

                adapter_config = os.path.join(init_ckpt, "adapter_config.json")
                if os.path.isfile(adapter_config):
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
                lora_source = init_ckpt if os.path.isfile(os.path.join(init_ckpt, "adapter_config.json")) else "new"
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
    gen_ids = out_ids[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


__all__ = [
    "MultimodalModel",
    "RECIPE_STAGE1",
    "RECIPE_STAGE2",
    "RECIPE_STAGE3",
    "generate_with_single_token_structure",
]
