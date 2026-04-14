"""
Unified Uni-Mol BOS + Qwen3 (4-bit) stack for Stage1 / Stage2 / downstream full SFT.

Recipes (string values):
  - stage1_frozen_llm: frozen LLM; train Uni-Mol + BosProjection only.
  - stage2_full_sft: full SFT (LoRA + Uni-Mol + BosProjection), init from Stage1 ``adapter_path``;
    forward supports mixed batches of LMDB (3D slot) and plain text (same data pipeline as Stage2.py).
  - stage3_full_sft: same full SFT stack; loads via ``init_ckpt`` / ``unimol_ckpt`` as used by Property / NiComplex / Vaska;
    forward is BOS 3D path only (no mixed batch).

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
    UNIMOL_ORIGIN,
    BosProjectionLayer,
    build_batch_multi,
    build_embeds_bos_only,
    build_embeds_text_only,
    extract_bos_repr,
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


def _resolve_dict_path(unimol_dict_path: Optional[str]) -> str:
    dict_path = unimol_dict_path or os.path.join(UNIMOL_ORIGIN, "unimol", "example_data", "molecule", "dict.txt")
    if not os.path.isfile(dict_path):
        dict_path = "/data/jingyuan_data/OMol25_MC/dict.txt"
    return dict_path


def _load_unimol_state(unimol_checkpoint: str):
    if os.path.isdir(unimol_checkpoint):
        sf_path = os.path.join(unimol_checkpoint, "model.safetensors")
        if os.path.isfile(sf_path):
            return load_file(sf_path)
        raise FileNotFoundError(f"No model.safetensors found under {unimol_checkpoint}")
    state = torch.load(unimol_checkpoint, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    return state


def _strip_module_prefix(state):
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}


class BosUnimolQwenModel(nn.Module):
    """Single implementation for BOS-slot 3D + Qwen; select behavior via ``recipe``."""

    def __init__(
        self,
        model_name: str,
        unimol_dict_path: str,
        *,
        recipe: str,
        unimol_checkpoint: Optional[str] = None,
        adapter_path: Optional[str] = None,
        unimol_ckpt: Optional[str] = None,
        init_ckpt: Optional[str] = None,
        dipole_mean: Optional[float] = None,
        dipole_std: Optional[float] = None,
        train_unimol: Optional[bool] = None,
        train_projection: Optional[bool] = None,
        train_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_target: str = "qv",
        use_bridge: bool = False,
        regression_mode: bool = False,
        include_bos: bool = True,
        bos_only: bool = True,
    ):
        super().__init__()
        if recipe not in (RECIPE_STAGE1, RECIPE_STAGE2, RECIPE_STAGE3):
            raise ValueError(f"Unknown recipe={recipe!r}")

        if recipe == RECIPE_STAGE3:
            if not bos_only or use_bridge or regression_mode:
                raise NotImplementedError(
                    "bos_unimol_qwen_model only implements the BOS-only generative path; "
                    "for bridge / regression use SFT_dipole_bridge_unimol_full.py."
                )
        self.recipe = recipe
        self.bos_only = True
        self.use_bridge = False
        self.include_bos = True
        self.regression_mode = False
        self.bridge_layer = None
        self.projection_layer = None
        self.regression_head = None

        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._lora_target_modules = _LORA_TARGET_MAP.get(lora_target, ["q_proj", "v_proj"])
        self.supports_mixed_batch = recipe == RECIPE_STAGE2
        self.dipole_mean = dipole_mean
        self.dipole_std = dipole_std
        self._train_unimol = train_unimol
        self._train_projection = train_projection
        self._train_lora = train_lora

        in_distributed = os.environ.get("LOCAL_RANK") is not None
        device_map = None if in_distributed else "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        from unicore.data import Dictionary

        dict_path = _resolve_dict_path(unimol_dict_path)
        self.dictionary = Dictionary.load(dict_path)
        self.dictionary.add_symbol("[MASK]", is_special=True)
        self._pad_idx = self.dictionary.pad()
        self._bos_idx = self.dictionary.bos()
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
            if not unimol_checkpoint:
                raise ValueError("stage1_frozen_llm requires unimol_checkpoint")
            state_to_load = _strip_module_prefix(_load_unimol_state(unimol_checkpoint))
            self.unimol.load_state_dict(state_to_load, strict=False)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"[BosUnimolQwen] stage1: Loaded Uni-Mol from {unimol_checkpoint}")
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
                print("[BosUnimolQwen] stage1: LLM frozen")

        elif recipe == RECIPE_STAGE2:
            if not adapter_path:
                raise ValueError("stage2_full_sft requires adapter_path (Stage1 BOS-only output directory)")
            unimol_pt = os.path.join(adapter_path, "unimol.pt")
            if os.path.isfile(unimol_pt):
                state = torch.load(unimol_pt, map_location="cpu", weights_only=False)
                if "model" in state:
                    state = state["model"]
                self.unimol.load_state_dict(state, strict=False)
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[BosUnimolQwen] stage2: Loaded Uni-Mol from {unimol_pt}")
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
                    print(f"[BosUnimolQwen] stage2: LoRA from {adapter_path}")
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
                        f"[BosUnimolQwen] stage2: new LoRA r={self._lora_r}, alpha={self._lora_alpha}, "
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
            self.bos_projection_layer = BosProjectionLayer(hidden_size).to(device=device, dtype=emb.dtype)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print("[BosUnimolQwen] stage3_full_sft: BOS -> BosProjection -> LLM")

            if init_ckpt and os.path.isdir(init_ckpt):
                unimol_pt = os.path.join(init_ckpt, "unimol.pt")
                unimol_loaded_from = None
                if os.path.isfile(unimol_pt):
                    state = torch.load(unimol_pt, map_location="cpu", weights_only=False)
                    if "model" in state:
                        state = state["model"]
                    self.unimol.load_state_dict(state, strict=False)
                    unimol_loaded_from = unimol_pt
                else:
                    default_unimol_ckpt = "/home/zhujingyuan/Uni-Mol/save/checkpoint_best.pt"
                    if os.path.isfile(default_unimol_ckpt):
                        state = torch.load(default_unimol_ckpt, map_location="cpu", weights_only=False)
                        if "model" in state:
                            state = state["model"]
                        self.unimol.load_state_dict(_strip_module_prefix(state), strict=False)
                        unimol_loaded_from = default_unimol_ckpt
                    elif int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"[BosUnimolQwen] Warning: {unimol_pt} and default Uni-Mol missing; random init")
                if unimol_loaded_from and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[BosUnimolQwen] Loaded Uni-Mol from {unimol_loaded_from}")

                adapter_config = os.path.join(init_ckpt, "adapter_config.json")
                if os.path.isfile(adapter_config):
                    self.llm = PeftModel.from_pretrained(llm, init_ckpt, is_trainable=True)
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"[BosUnimolQwen] Loaded LoRA from {init_ckpt}")
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
                        print(f"[BosUnimolQwen] Created new LoRA (r={self._lora_r})")
            else:
                loaded_unimol = False
                if unimol_ckpt:
                    if os.path.isdir(unimol_ckpt):
                        safetensor_path = os.path.join(unimol_ckpt, "model.safetensors")
                        if os.path.isfile(safetensor_path):
                            state = load_file(safetensor_path, device="cpu")
                            self.unimol.load_state_dict(_strip_module_prefix(state), strict=False)
                            loaded_unimol = True
                            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                                print(f"[BosUnimolQwen] Loaded Uni-Mol from {safetensor_path}")
                    elif os.path.isfile(unimol_ckpt):
                        if unimol_ckpt.endswith(".safetensors"):
                            state = load_file(unimol_ckpt, device="cpu")
                        else:
                            state = torch.load(unimol_ckpt, map_location="cpu", weights_only=False)
                            if "model" in state:
                                state = state["model"]
                        self.unimol.load_state_dict(_strip_module_prefix(state), strict=False)
                        loaded_unimol = True
                        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                            print(f"[BosUnimolQwen] Loaded Uni-Mol from {unimol_ckpt}")
                if not loaded_unimol and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print("[BosUnimolQwen] Uni-Mol: no valid checkpoint; random init")

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
                    print(f"[BosUnimolQwen] Created new LoRA (r={self._lora_r})")

            self.llm.print_trainable_parameters()

            if init_ckpt and os.path.isdir(init_ckpt):
                unimol_pt_in_ckpt = os.path.join(init_ckpt, "unimol.pt")
                default_unimol_ckpt = "/home/zhujingyuan/Uni-Mol/save/checkpoint_best.pt"
                if os.path.isfile(unimol_pt_in_ckpt):
                    unimol_source = unimol_pt_in_ckpt
                elif os.path.isfile(default_unimol_ckpt):
                    unimol_source = default_unimol_ckpt
                else:
                    unimol_source = "random_init"
                lora_source = init_ckpt if os.path.isfile(os.path.join(init_ckpt, "adapter_config.json")) else "new"
            else:
                unimol_source = unimol_ckpt if unimol_ckpt else "random_init"
                lora_source = "new"

            if self._train_unimol is None:
                self._train_unimol = not (init_ckpt and os.path.isdir(init_ckpt))
            if self._train_projection is None:
                self._train_projection = not (init_ckpt and os.path.isdir(init_ckpt))

            for p in self.unimol.parameters():
                p.requires_grad = self._train_unimol
            if not self._train_lora:
                for n, p in self.llm.named_parameters():
                    if "lora" in n.lower():
                        p.requires_grad = False

            bos_proj_source = "new"
            if init_ckpt and os.path.isdir(init_ckpt):
                bos_safetensors = os.path.join(init_ckpt, "bos_projection.safetensors")
                bos_pt = os.path.join(init_ckpt, "bos_projection.pt")
                if os.path.isfile(bos_safetensors):
                    bos_state = load_file(bos_safetensors, device="cpu")
                    self.bos_projection_layer.load_state_dict(bos_state, strict=False)
                    bos_proj_source = bos_safetensors
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"[BosUnimolQwen] Loaded BOS projection from {bos_safetensors}")
                elif os.path.isfile(bos_pt):
                    bos_state = torch.load(bos_pt, map_location="cpu", weights_only=False)
                    self.bos_projection_layer.load_state_dict(bos_state, strict=False)
                    bos_proj_source = bos_pt
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"[BosUnimolQwen] Loaded BOS projection from {bos_pt}")
                elif int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print("[BosUnimolQwen] BosProjection trained from scratch")

            train_bos = self._train_projection if self._train_projection is not None else True
            for p in self.bos_projection_layer.parameters():
                p.requires_grad = train_bos

            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                self._print_module_config_table(unimol_source, lora_source, bos_proj_source)

        if recipe != RECIPE_STAGE3:
            device = next(self.llm.parameters()).device
            with torch.no_grad():
                dummy = torch.tensor([[self.tokenizer.eos_token_id]], device=device)
                emb = self.llm.get_input_embeddings()(dummy)
            hidden_size = emb.shape[-1]
            self.bos_projection_layer = BosProjectionLayer(hidden_size).to(device=device, dtype=emb.dtype)
            if recipe == RECIPE_STAGE2:
                bos_proj_path = os.path.join(adapter_path, "bos_projection.safetensors")
                if os.path.isfile(bos_proj_path):
                    st = load_file(bos_proj_path)
                    bos_state = {k.replace("bos_projection.", ""): v for k, v in st.items() if k.startswith("bos_projection.")}
                    if bos_state:
                        self.bos_projection_layer.load_state_dict(bos_state, strict=False)
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"[BosUnimolQwen] Loaded BosProjection from {bos_proj_path}")
                else:
                    bos_pt = os.path.join(adapter_path, "bos_projection.pt")
                    if os.path.isfile(bos_pt):
                        st = torch.load(bos_pt, map_location="cpu", weights_only=False)
                        self.bos_projection_layer.load_state_dict(st, strict=False)
                        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                            print(f"[BosUnimolQwen] Loaded BosProjection from {bos_pt}")
                for p in self.bos_projection_layer.parameters():
                    p.requires_grad = True
            else:
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[BosUnimolQwen] stage1: BosProjection({ATOM_DIM}->{hidden_size})")

        self._start_3d_id = self.tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
        self._end_3d_id = self.tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
        self.unimol = self.unimol.to(next(self.llm.parameters()).device)

    def _print_module_config_table(self, unimol_source, lora_source, bos_proj_source="new"):
        def count_params(module):
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters())

        unimol_params = count_params(self.unimol)
        bos_proj_params = count_params(self.bos_projection_layer)
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

        train_bos = self._train_projection if self._train_projection is not None else True
        print("\n" + "=" * 90)
        print("                         Module config (BOS-only)")
        print("=" * 90)
        print(f"{'Module':<20} {'Train':<12} {'#Params':<15} {'Source'}")
        print("-" * 90)
        print(f"{'Uni-Mol':<20} {'train' if self._train_unimol else 'frozen':<12} {fmt_params(unimol_params):<15} {unimol_source}")
        print(f"{'BosProjection':<20} {'train' if train_bos else 'frozen':<12} {fmt_params(bos_proj_params):<15} {bos_proj_source}")
        print(f"{'LoRA':<20} {'train' if self._train_lora else 'frozen':<12} {fmt_params(lora_params):<15} {lora_source}")
        print("-" * 90)
        total_trainable = 0
        if self._train_unimol:
            total_trainable += unimol_params
        if train_bos:
            total_trainable += bos_proj_params
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
            bos_idx=self._bos_idx,
            eos_idx=self._eos_idx,
            device=str(device),
        )
        encoder_rep, _ = self.unimol(
            batch_dict["src_tokens"],
            batch_dict["src_distance"],
            batch_dict["src_coord"],
            batch_dict["src_edge_type"],
        )
        bos_repr = extract_bos_repr(encoder_rep)
        return build_embeds_bos_only(
            self.llm,
            device,
            before_3d_ids,
            before_3d_mask,
            after_3d_ids,
            after_3d_mask,
            response_ids,
            response_mask,
            bos_repr,
            self.bos_projection_layer,
            self._start_3d_id,
            self._end_3d_id,
        )

    def _get_unimol_bos(self, list_atoms, list_coordinates, device):
        batch_dict = build_batch_multi(
            list_atoms,
            list_coordinates,
            self.dictionary,
            max_seq_len=UNIMOL_MAX_SEQ_LEN,
            pad_idx=self._pad_idx,
            bos_idx=self._bos_idx,
            eos_idx=self._eos_idx,
            device=str(device),
        )
        encoder_rep, _ = self.unimol(
            batch_dict["src_tokens"],
            batch_dict["src_distance"],
            batch_dict["src_coord"],
            batch_dict["src_edge_type"],
        )
        return extract_bos_repr(encoder_rep)

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
            bos_repr = self._get_unimol_bos(list_atoms, list_coordinates, device)
            inputs_embeds, attention_mask, labels = build_embeds_bos_only(
                self.llm,
                device,
                before_3d_ids,
                before_3d_mask,
                after_3d_ids,
                after_3d_mask,
                response_ids,
                response_mask,
                bos_repr,
                self.bos_projection_layer,
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
                bos_idx=self._bos_idx,
                eos_idx=self._eos_idx,
                device=str(device),
            )
            encoder_rep, _ = self.unimol(
                batch_dict["src_tokens"],
                batch_dict["src_distance"],
                batch_dict["src_coord"],
                batch_dict["src_edge_type"],
            )
            bos_repr = extract_bos_repr(encoder_rep)
            inputs_embeds, attention_mask, labels = build_embeds_bos_only(
                self.llm,
                device,
                before_3d_ids,
                before_3d_mask,
                after_3d_ids,
                after_3d_mask,
                response_ids,
                response_mask,
                bos_repr,
                self.bos_projection_layer,
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


__all__ = [
    "BosUnimolQwenModel",
    "RECIPE_STAGE1",
    "RECIPE_STAGE2",
    "RECIPE_STAGE3",
]
