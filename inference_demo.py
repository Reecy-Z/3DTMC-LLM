"""
Generic single-token + 3D encoder chat demo: user provides SMILES, an XYZ path, and any instruction.

3D structure is encoded with the 3D encoder and injected at the object_ref slot; the LLM sees your
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

from Stage3.Property import PROPERTY_CONFIG
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel, generate_with_single_token_structure
from train_defaults import STAGE2_DEFAULTS
from transformers import AutoTokenizer
from utils import (
    format_instruction_field,
    _atoms_coords_remove_h_center,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XYZ = os.path.join(_DIR, "CC(C)P(->[Au+]<-[I-])(C(C)C)C(C)C.xyz")
DEFAULT_SMILES = "CC(C)P(->[Au+]<-[I-])(C(C)C)C(C)C"
DEFAULT_INSTRUCTION = "Give a description of this transition metal complex:"


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


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalModel(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        init_ckpt=args.stage2_ckpt,
        train_3d_encoder=False,
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
    p.add_argument("--model_name", type=str, default=STAGE2_DEFAULTS["model_name"])
    p.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=STAGE2_DEFAULTS["3D_encoder_dict"],
    )
    p.add_argument("--Stage2_ckpt", dest="stage2_ckpt", type=str, default=STAGE2_DEFAULTS["output_dir"])
    p.add_argument("--lora_r", type=int, default=STAGE2_DEFAULTS["lora_r"])
    p.add_argument("--lora_alpha", type=int, default=STAGE2_DEFAULTS["lora_alpha"])
    p.add_argument("--lora_target", type=str, default=STAGE2_DEFAULTS["lora_target"])
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
        help="Path to XYZ file with the 3D structure (3D encoder input).",
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
        description="Generic demo: SMILES + XYZ + free-form instruction with single-token 3D conditioning.",
    )
    add_demo_args(parser)
    args = parser.parse_args()

    if not args.stage2_ckpt or not os.path.isdir(args.stage2_ckpt):
        print("--Stage2_ckpt must be a directory with adapter + 3D_encoder.pt + single_token_projection.*", file=sys.stderr)
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
        text = generate_with_single_token_structure(
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
