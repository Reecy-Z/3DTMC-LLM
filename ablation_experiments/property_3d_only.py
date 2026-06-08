"""
Property ablation: standard Uni-Mol [CLS] -> single-token projection -> Qwen3,
with NO SMILES in the text prompt (3D structure slot only).

Trains single_token_projection.pt + LoRA + 3D encoder from scratch by default
(same fair-init policy as property_k8.py).

Example:
  deepspeed --num_gpus=2 ablation_experiments/property_3d_only.py \\
    --3D_encoder_ckpt /data/jingyuan_data/checkpoint-unimol-OMol-all-500000 \\
    --property dipole_moment
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
from utils import THREE_D_ENCODER_STATE_PT, MultimodalCollator, MultimodalFullTrainer

from Property import PROPERTY_CONFIG
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel
from train_defaults import PROPERTY_DEFAULTS, VASKA_DEFAULTS

from ablation_experiments.dataset_3d_only import TmQMg3DOnlyUnimolDataset

TRAIN_LMDB = "/data/jingyuan_data/tmqmg/stage3/train/tmqmg_atom_only_new.lmdb"

PROPERTY_3D_ONLY_DEFAULTS = {
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
        description="Property ablation: single-token 3D slot only (no SMILES in prompt)"
    )
    parser.add_argument("--model_name", type=str, default=PROPERTY_3D_ONLY_DEFAULTS["model_name"])
    parser.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=PROPERTY_3D_ONLY_DEFAULTS["3D_encoder_dict"],
    )
    parser.add_argument(
        "--3D_encoder_ckpt",
        dest="three_d_encoder_ckpt",
        type=str,
        default=PROPERTY_DEFAULTS["3D_encoder_ckpt"],
    )
    parser.add_argument("--Stage2_ckpt", dest="stage2_ckpt", type=str, default=PROPERTY_DEFAULTS["Stage2_ckpt"])
    parser.add_argument("--train_lmdb", type=str, default=PROPERTY_3D_ONLY_DEFAULTS["train_lmdb"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument(
        "--property",
        type=str,
        default=PROPERTY_3D_ONLY_DEFAULTS["property"],
        choices=["dipole_moment", "polarisability", "homo_lumo_gap"],
    )
    parser.add_argument("--lora_r", type=int, default=PROPERTY_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=PROPERTY_DEFAULTS["lora_alpha"])
    parser.add_argument("--lora_target", type=str, default=PROPERTY_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"])
    parser.add_argument("--lora_init", type=str, default="from_scratch", choices=["from_scratch", "pretrained"])
    parser.add_argument(
        "--train_3d_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train Uni-Mol 3D encoder (default on).",
    )
    parser.add_argument(
        "--train_projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train single_token_projection (default on; init from scratch unless --lora_init pretrained).",
    )
    parser.add_argument(
        "--train_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train LoRA on Qwen (default on). Base weights stay frozen (4bit).",
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

    _require_existing_path(args_main.model_name, "--model_name")
    encoder_ckpt = _resolve_three_d_encoder_ckpt(args_main.three_d_encoder_ckpt, args_main.stage2_ckpt)
    init_ckpt = args_main.stage2_ckpt if args_main.lora_init == "pretrained" else None

    if local_rank == 0:
        wandb.init(project=f"Stage3_Property_{prop_cfg['output_dir_suffix']}_3d_only")
        print(
            f"[Property-3D-only] property={args_main.property}; "
            f"single-token [CLS]; lora_init={args_main.lora_init}; "
            f"prompt=instruction only (NO SMILES)"
        )
        print(f"[Property-3D-only] 3D encoder weights: {encoder_ckpt}")
        print(f"[Property-3D-only] Qwen base (frozen 4bit): {args_main.model_name}")
        print(
            f"[Property-3D-only] trainable: 3D_encoder={args_main.train_3d_encoder}, "
            f"projection={args_main.train_projection}, lora={args_main.train_lora}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = MultimodalModel(
        args_main.model_name,
        args_main.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=encoder_ckpt,
        init_ckpt=init_ckpt,
        train_3d_encoder=args_main.train_3d_encoder,
        train_projection=args_main.train_projection,
        train_lora=args_main.train_lora,
        load_pretrained_projection=False,
        load_pretrained_lora=(args_main.lora_init == "pretrained"),
        lora_r=args_main.lora_r,
        lora_alpha=args_main.lora_alpha,
        lora_target=args_main.lora_target,
    )
    if local_rank == 0:
        if args_main.train_lora:
            model.llm.print_trainable_parameters()
        else:
            print("[Property-3D-only] LoRA frozen")

    train_dataset = TmQMg3DOnlyUnimolDataset(
        [args_main.train_lmdb],
        tokenizer=tokenizer,
        property_key=prop_key,
    )
    if local_rank == 0:
        print(f"[Property-3D-only] Train samples: {len(train_dataset)} (no validation)")

    output_dir = args_main.output_dir or (
        f"/data/jingyuan_data/Stage3_Property_{prop_cfg['output_dir_suffix']}_3d_only_ckpt"
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
    trainer = MultimodalFullTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()
