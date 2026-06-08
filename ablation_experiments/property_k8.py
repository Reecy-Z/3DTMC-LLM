"""
Property ablation: Uni-Mol atom hiddens -> 8 learnable query tokens (cross-attn) -> Qwen3.

Does not load single_token_projection.pt or Stage2 LoRA by default:
trains k8_projection.pt and a new LoRA on base Qwen from scratch.
Provide the 3D encoder via --3D_encoder_ckpt (file or dir), or --Stage2_ckpt as a directory
that contains 3D_encoder.pt (convenience alias; does not load Stage2 LoRA by default).

Example:
  deepspeed --num_gpus=2 ablation_experiments/property_k8.py \\
    --3D_encoder_ckpt /path/to/Stage2/3D_encoder.pt --property dipole_moment
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_STAGE3_DIR = os.path.join(_PROJECT_ROOT, "Stage3")
if _STAGE3_DIR not in sys.path:
    sys.path.insert(0, _STAGE3_DIR)

import utils  # noqa: F401
import wandb
from transformers import AutoTokenizer, TrainingArguments
from utils import THREE_D_ENCODER_STATE_PT, MultimodalCollator

from Property import PROPERTY_CONFIG, TmQMgSingleTokenUnimolDataset
from multimodal_LLM import RECIPE_STAGE3
from train_defaults import PROPERTY_DEFAULTS, VASKA_DEFAULTS

from ablation_experiments.multimodal_k8 import MultimodalK8Trainer, MultimodalModelK8

TRAIN_LMDB = "/data/jingyuan_data/tmqmg/stage3/train/tmqmg_atom_only_new.lmdb"

# Overrides PROPERTY_DEFAULTS placeholders for local runs (see train_defaults.VASKA_DEFAULTS).
PROPERTY_K8_DEFAULTS = {
    **PROPERTY_DEFAULTS,
    "model_name": VASKA_DEFAULTS["model_name"],
    "3D_encoder_dict": VASKA_DEFAULTS["3D_encoder_dict"],
    "train_lmdb": TRAIN_LMDB,
    "property": "dipole_moment",
}


def _require_existing_path(path, flag_name):
    if not path or "/path/to/" in path or not os.path.isdir(path):
        raise FileNotFoundError(
            f"{flag_name} must be a local directory with the HF model, got {path!r}. "
            f"Example: --model_name {VASKA_DEFAULTS['model_name']}"
        )


def _resolve_three_d_encoder_ckpt(three_d_encoder_ckpt, stage2_ckpt_dir):
    """Resolve Uni-Mol weights path. Stage2_ckpt is only used to find 3D_encoder.pt."""
    if three_d_encoder_ckpt:
        return three_d_encoder_ckpt
    if stage2_ckpt_dir:
        path = os.path.join(stage2_ckpt_dir, THREE_D_ENCODER_STATE_PT)
        if os.path.isfile(path) or os.path.isdir(path):
            return path
        raise FileNotFoundError(
            f"--Stage2_ckpt={stage2_ckpt_dir!r} has no {THREE_D_ENCODER_STATE_PT}; "
            "pass --3D_encoder_ckpt explicitly."
        )
    raise FileNotFoundError(
        "3D encoder checkpoint required: set --3D_encoder_ckpt or --Stage2_ckpt "
        f"(directory containing {THREE_D_ENCODER_STATE_PT})."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Property ablation: 8 learnable structure query tokens + Qwen3 generative SFT"
    )
    parser.add_argument("--model_name", type=str, default=PROPERTY_K8_DEFAULTS["model_name"])
    parser.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=PROPERTY_K8_DEFAULTS["3D_encoder_dict"],
    )
    parser.add_argument(
        "--3D_encoder_ckpt",
        dest="three_d_encoder_ckpt",
        type=str,
        default=PROPERTY_DEFAULTS["3D_encoder_ckpt"],
        help="Uni-Mol weights: .pt file, model.safetensors dir, or None to use Stage2_ckpt/3D_encoder.pt",
    )
    parser.add_argument(
        "--Stage2_ckpt",
        dest="stage2_ckpt",
        type=str,
        default=PROPERTY_DEFAULTS["Stage2_ckpt"],
        help=(
            "Optional Stage2 output directory; default use is ONLY "
            f"{THREE_D_ENCODER_STATE_PT} inside it. Full Stage2 (LoRA) loads only if --lora_init pretrained."
        ),
    )
    parser.add_argument("--train_lmdb", type=str, default=PROPERTY_K8_DEFAULTS["train_lmdb"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N training steps")
    parser.add_argument(
        "--property",
        type=str,
        default=PROPERTY_K8_DEFAULTS["property"],
        choices=["dipole_moment", "polarisability", "homo_lumo_gap"],
    )
    parser.add_argument("--lora_r", type=int, default=PROPERTY_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=PROPERTY_DEFAULTS["lora_alpha"])
    parser.add_argument("--lora_target", type=str, default=PROPERTY_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"])
    parser.add_argument(
        "--lora_init",
        type=str,
        default="from_scratch",
        choices=["from_scratch", "pretrained"],
        help="from_scratch: new LoRA on base Qwen; pretrained: load Stage2 adapter weights",
    )
    parser.add_argument(
        "--train_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train LoRA on Qwen (default on). Base weights stay frozen (4bit). Use --no-train_lora to freeze LLM side.",
    )
    parser.add_argument("--epochs", type=int, default=PROPERTY_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=PROPERTY_DEFAULTS["lr"])
    parser.add_argument("--batch_size", type=int, default=PROPERTY_DEFAULTS["batch_size"])
    parser.add_argument("--local_rank", type=int, default=-1)
    args_main = parser.parse_args()

    try:
        import multiprocessing

        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    prop_cfg = PROPERTY_CONFIG[args_main.property]
    prop_key = prop_cfg["key"]
    # Same prompt style as dipole: instruction + SMILES only (no polished_description).
    instruction_use = prop_cfg.get("instruction_smiles") or prop_cfg.get("instruction_description")

    _require_existing_path(args_main.model_name, "--model_name")
    encoder_ckpt = _resolve_three_d_encoder_ckpt(args_main.three_d_encoder_ckpt, args_main.stage2_ckpt)
    init_ckpt = args_main.stage2_ckpt if args_main.lora_init == "pretrained" else None

    if local_rank == 0:
        wandb.init(project=f"Stage3_Property_{prop_cfg['output_dir_suffix']}_k8")
        print(
            f"[Property-K8] property={args_main.property}, "
            f"8 learnable query tokens; lora_init={args_main.lora_init}; train_lora={args_main.train_lora}; "
            f"prompt=instruction+SMILES (no polished_description)"
        )
        print(f"[Property-K8] 3D encoder weights: {encoder_ckpt}")
        print(f"[Property-K8] Qwen base (frozen 4bit): {args_main.model_name}")
        if args_main.train_lora:
            print("[Property-K8] LLM: LoRA adapters are TRAINABLE")
        else:
            print("[Property-K8] LLM: LoRA frozen (only k8 projector + optional 3D encoder train)")

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = MultimodalModelK8(
        args_main.model_name,
        args_main.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=encoder_ckpt,
        init_ckpt=init_ckpt,
        train_3d_encoder=True,
        train_projection=True,
        train_lora=args_main.train_lora,
        lora_r=args_main.lora_r,
        lora_alpha=args_main.lora_alpha,
        lora_target=args_main.lora_target,
        load_pretrained_lora=(args_main.lora_init == "pretrained"),
    )
    if local_rank == 0 and args_main.train_lora:
        model.llm.print_trainable_parameters()
    train_dataset = TmQMgSingleTokenUnimolDataset(
        [args_main.train_lmdb],
        tokenizer=tokenizer,
        property_key=prop_key,
        instruction=instruction_use,
        use_polished_description=False,
    )
    if local_rank == 0:
        print(f"[Property-K8] Train samples: {len(train_dataset)} (no validation)")

    output_dir = args_main.output_dir or (
        f"/data/jingyuan_data/Stage3_Property_{prop_cfg['output_dir_suffix']}_k8_ckpt"
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args_main.epochs,
        deepspeed=os.path.join(_PROJECT_ROOT, "ds_config.json"),
        per_device_train_batch_size=args_main.batch_size,
        per_device_eval_batch_size=args_main.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args_main.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        save_steps=args_main.save_steps,
        save_strategy="steps",
        eval_strategy="no",
        report_to="wandb",
        ddp_find_unused_parameters=False,
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        save_total_limit=20,
    )
    trainer = MultimodalK8Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()
