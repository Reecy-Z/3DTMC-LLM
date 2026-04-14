"""
Generic BOS + Uni-Mol chat demo: user provides SMILES, an XYZ path, and any instruction.

3D structure is encoded with Uni-Mol and injected at the object_ref slot; the LLM sees your
text (instruction + optional SMILES / description) together with that 3D token.

Default instruction is the HOMO–LUMO example from ``Property.PROPERTY_CONFIG`` (line ~48);
replace it with any question or instruction you like. Checkpoint quality still depends on what
the model was fine-tuned on.

Examples:
  CUDA_VISIBLE_DEVICES=0 python inference_demo.py \\
    --smiles "CC(C)P(->[Au+]<-[I-])(C(C)C)C(C)C" \\
    --xyz "./CC(C)P(->[Au+]<-[I-])(C(C)C)C(C)C.xyz" \\
    --instruction "What is the HOMO-LUMO gap (in Ha) of this transition metal complex? \\
Given the description, SMILES and structure, respond with the numerical value only:"

  CUDA_VISIBLE_DEVICES=0 python inference_demo.py --interactive
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

import utils  # noqa: F401

from Property import PROPERTY_CONFIG
from bos_unimol_qwen_model import BosUnimolQwenModel, RECIPE_STAGE3
from train_defaults import PROPERTY_DEFAULTS
from transformers import AutoTokenizer
from utils import (
    UNIMOL_MAX_SEQ_LEN,
    OBJECT_REF_CHAT_SEP,
    build_batch_multi,
    extract_bos_repr,
    format_instruction_field,
    _atoms_coords_remove_h_center,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XYZ = os.path.join(_DIR, "CC(C)P(->[Au+]<-[I-])(C(C)C)C(C)C.xyz")
DEFAULT_SMILES = "CC(C)P(->[Au+]<-[I-])(C(C)C)C(C)C"
# Same text as Property.py ``homo_lumo_gap`` → ``instruction_description`` (example default only).
DEFAULT_INSTRUCTION = PROPERTY_CONFIG["homo_lumo_gap"]["instruction_description"]


def read_xyz(path: str):
    """Read a simple XYZ file: line0 = n_atoms, line1 = comment, then element x y z."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms, coords = [], []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        if len(parts) < 4:
            continue
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.asarray(coords, dtype=np.float32)


def build_user_content(
    instruction: str,
    smiles: str,
    *,
    description: str | None = None,
    append_smiles: bool = True,
) -> str:
    instruction = (instruction or "").strip()
    smiles = format_instruction_field(smiles)
    if description and str(description).strip():
        body = instruction
        if append_smiles and smiles:
            body = f"{instruction} {smiles}"
        return f"{str(description).strip()}\n{body}"
    if append_smiles and smiles:
        return f"{instruction} {smiles}"
    return instruction


def generate_with_bos_structure(
    model: BosUnimolQwenModel,
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
        bos_idx=model._bos_idx,
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
    bos_repr = extract_bos_repr(encoder_rep)
    proj_dtype = next(model.bos_projection_layer.parameters()).dtype
    bos_repr = bos_repr.to(dtype=proj_dtype)
    with torch.inference_mode():
        mol_embeds = model.bos_projection_layer(bos_repr).unsqueeze(1)

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


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    unimol_ckpt = args.unimol_ckpt or None
    model = BosUnimolQwenModel(
        args.model_name,
        args.unimol_dict,
        recipe=RECIPE_STAGE3,
        unimol_ckpt=unimol_ckpt,
        init_ckpt=args.init_ckpt,
        train_unimol=False,
        train_projection=False,
        train_lora=False,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
    )
    model.eval()
    return model, tokenizer


def prepare_geometry(xyz_path: str):
    if not os.path.isfile(xyz_path):
        raise FileNotFoundError(f"XYZ not found: {xyz_path}")
    atoms, coords = read_xyz(xyz_path)
    atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
    if not atoms:
        raise RuntimeError("No heavy atoms left after H-strip / centering.")
    return atoms, coords


def add_demo_args(p: argparse.ArgumentParser):
    p.add_argument("--model_name", type=str, default=PROPERTY_DEFAULTS["model_name"])
    p.add_argument("--unimol_dict", type=str, default=PROPERTY_DEFAULTS["unimol_dict"])
    p.add_argument("--unimol_ckpt", type=str, default=PROPERTY_DEFAULTS["unimol_ckpt"] or "")
    p.add_argument("--init_ckpt", type=str, default=PROPERTY_DEFAULTS["init_ckpt"])
    p.add_argument("--lora_r", type=int, default=PROPERTY_DEFAULTS["lora_r"])
    p.add_argument("--lora_alpha", type=int, default=PROPERTY_DEFAULTS["lora_alpha"])
    p.add_argument("--lora_target", type=str, default=PROPERTY_DEFAULTS["lora_target"])
    p.add_argument(
        "--smiles",
        type=str,
        default=DEFAULT_SMILES,
        help="SMILES string for the complex (printed in the user message unless --no_append_smiles).",
    )
    p.add_argument(
        "--xyz",
        type=str,
        default=DEFAULT_XYZ,
        help="Path to XYZ file with the 3D structure (Uni-Mol input).",
    )
    p.add_argument(
        "--instruction",
        "-i",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Your question or instruction (any task). Default is the HOMO–LUMO Property example.",
    )
    p.add_argument(
        "--instruction_file",
        type=str,
        default="",
        help="If set, read instruction text from this file (overrides --instruction).",
    )
    p.add_argument(
        "--description",
        type=str,
        default="",
        help="Optional leading paragraph (like polished_description); prepended before instruction block.",
    )
    p.add_argument(
        "--no_append_smiles",
        action="store_true",
        help="Do not append SMILES after the instruction; use --instruction as the full user text (you can embed SMILES yourself).",
    )
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--do_sample", action="store_true", help="Sample instead of greedy decode (more 'chatty').")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument(
        "--interactive",
        action="store_true",
        help="After loading the model, repeatedly ask for SMILES, XYZ path, and instruction until you type 'q'.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generic demo: SMILES + XYZ + free-form instruction with BOS 3D conditioning.",
    )
    add_demo_args(parser)
    args = parser.parse_args()

    if not args.init_ckpt or not os.path.isdir(args.init_ckpt):
        print("--init_ckpt must be a directory with adapter + unimol.pt + bos_projection.*", file=sys.stderr)
        sys.exit(1)

    instruction = args.instruction
    if args.instruction_file:
        with open(args.instruction_file, "r", encoding="utf-8", errors="replace") as f:
            instruction = f.read().strip()
        if not instruction:
            print("--instruction_file is empty", file=sys.stderr)
            sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(args)

    def one_run(smiles: str, xyz_path: str, instr: str, description: str):
        user_content = build_user_content(
            instr,
            smiles,
            description=description.strip() or None,
            append_smiles=not args.no_append_smiles,
        )
        atoms, coords = prepare_geometry(xyz_path)
        text = generate_with_bos_structure(
            model,
            tokenizer,
            atoms,
            coords,
            user_content,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("---")
        print("user_content:\n", user_content)
        print("---")
        print("assistant:\n", text)

    if args.interactive:
        print("Interactive mode: enter SMILES, XYZ path, then instruction.")
        print("Empty line = keep previous value. Type q at SMILES prompt to quit.\n")
        cur_smiles = args.smiles.strip()
        cur_xyz = args.xyz
        cur_instr = instruction
        cur_desc = args.description
        while True:
            s = input(f"SMILES [{cur_smiles}]: ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            if s:
                cur_smiles = s
            z = input(f"XYZ path [{cur_xyz}]: ").strip()
            if z:
                cur_xyz = z
            print(f"Instruction [default length {len(cur_instr)} chars; Enter to keep]")
            t = input("> ").strip()
            if t:
                cur_instr = t
            d = input("Optional description (Enter to keep; '-' to clear): ").strip()
            if d == "-":
                cur_desc = ""
            elif d:
                cur_desc = d
            try:
                one_run(cur_smiles, cur_xyz, cur_instr, cur_desc)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"Error: {e}", file=sys.stderr)
        return

    one_run(args.smiles.strip(), args.xyz, instruction, args.description)


if __name__ == "__main__":
    main()
