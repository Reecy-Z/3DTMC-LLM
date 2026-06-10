"""
NiComplex OOD training: ligand-scaffold LOLO (Pybox / Biox / Biim) and reaction-type OOD.

Same model stack as Stage3/NiComplex.py (single-token + SMILES + temp + 3D → Qwen3 SFT).

Example (one experiment):
  deepspeed --num_gpus=2 OOD/NiComplex/NiComplex_OOD.py --experiment train_rest_test_Pybox

Example (all five experiments sequentially):
  deepspeed --num_gpus=2 OOD/NiComplex/NiComplex_OOD.py --run_all_experiments
"""
from __future__ import annotations

import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_STAGE3_DIR = os.path.join(_PROJECT_ROOT, "Stage3")
if _STAGE3_DIR not in sys.path:
    sys.path.insert(0, _STAGE3_DIR)

import utils  # noqa: F401
import wandb
from transformers import AutoTokenizer, TrainingArguments
from utils import MultimodalCollator, MultimodalFullTrainer

from NiComplex import NI_INSTRUCTION, NiComplexDDGDataset, load_merged_valid_nicomplex_records
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel
from train_defaults import NICOMPLEX_OOD_DEFAULTS, VASKA_DEFAULTS

from OOD.NiComplex.nicomplex_split import (
    EXPERIMENT_NAMES,
    experiment_dirname,
    split_by_experiment,
    summarize_field,
)

DEFAULT_LMDB = NICOMPLEX_OOD_DEFAULTS["lmdb"][0]
DEFAULT_OUTPUT_DIR = NICOMPLEX_OOD_DEFAULTS["output_dir"]
DEFAULT_BATCH_SIZE = NICOMPLEX_OOD_DEFAULTS["batch_size"]
DEFAULT_EPOCHS = NICOMPLEX_OOD_DEFAULTS["epochs"]
DEFAULT_SAVE_STEPS = NICOMPLEX_OOD_DEFAULTS["save_steps"]


def _train_one_experiment(args, experiment_name: str, all_valid: list, local_rank: int) -> None:
    train_samples, ood_test_samples, spec = split_by_experiment(all_valid, experiment_name)
    exp_dir = os.path.join(args.output_dir, experiment_dirname(experiment_name))

    if local_rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "ood_split.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment": experiment_name,
                    "split_col": spec["split_col"],
                    "train_types": sorted(spec["train_types"]),
                    "test_types": sorted(spec["test_types"]),
                    "description": spec["description"],
                    "R_Type_counts": summarize_field(all_valid, "R_Type"),
                    "L_Scaffold_counts": summarize_field(all_valid, "L_Scaffold"),
                    "n_train": len(train_samples),
                    "n_ood_test": len(ood_test_samples),
                },
                f,
                indent=2,
            )
        print(
            f"\n[NiComplex-OOD] ===== experiment={experiment_name} =====\n"
            f"  split_col={spec['split_col']}\n"
            f"  train={len(train_samples)} ood_test={len(ood_test_samples)}\n"
            f"  output_dir={exp_dir}"
        )
        if os.environ.get("WANDB_MODE", "").lower() not in ("disabled", "offline"):
            wandb.init(
                project="NiComplex_OOD",
                name=experiment_name,
                reinit=True,
                config={**vars(args), "experiment": experiment_name},
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalModel(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=args.three_d_encoder_ckpt,
        init_ckpt=args.stage2_ckpt,
        load_pretrained_projection=(args.projection_init == "pretrained"),
        train_3d_encoder=True,
        train_projection=True,
        train_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
    )

    train_dataset = NiComplexDDGDataset(
        tokenizer=tokenizer,
        samples=train_samples,
        instruction=NI_INSTRUCTION,
    )

    training_args = TrainingArguments(
        output_dir=exp_dir,
        num_train_epochs=args.epochs,
        deepspeed=os.path.join(_PROJECT_ROOT, "ds_config.json"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.05,
        logging_steps=10,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="no",
        report_to="none" if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "offline") else "wandb",
        ddp_find_unused_parameters=False,
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        save_total_limit=5,
    )

    trainer = MultimodalFullTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(exp_dir)

    if local_rank == 0:
        with open(os.path.join(exp_dir, "experiment.txt"), "w", encoding="utf-8") as f:
            f.write(experiment_name + "\n")
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NiComplex OOD: Pybox / Biox / Biim scaffold holdouts")
    parser.add_argument("--model_name", type=str, default=VASKA_DEFAULTS["model_name"])
    parser.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=VASKA_DEFAULTS["3D_encoder_dict"],
    )
    parser.add_argument("--Stage2_ckpt", dest="stage2_ckpt", type=str, default=VASKA_DEFAULTS["Stage2_ckpt"])
    parser.add_argument(
        "--3D_encoder_ckpt",
        dest="three_d_encoder_ckpt",
        type=str,
        default=NICOMPLEX_OOD_DEFAULTS["3D_encoder_ckpt"],
    )
    parser.add_argument("--lmdb", action="append", default=None, dest="lmdb_paths", metavar="PATH")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment", type=str, default=None, choices=list(EXPERIMENT_NAMES))
    parser.add_argument(
        "--run_all_experiments",
        action="store_true",
        help="Sequentially train all three scaffold OOD experiments",
    )
    parser.add_argument("--lora_r", type=int, default=NICOMPLEX_OOD_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=NICOMPLEX_OOD_DEFAULTS["lora_alpha"])
    parser.add_argument("--lora_target", type=str, default=NICOMPLEX_OOD_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"])
    parser.add_argument(
        "--projection_init",
        type=str,
        default=NICOMPLEX_OOD_DEFAULTS["projection_init"],
        choices=["pretrained", "from_scratch"],
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=NICOMPLEX_OOD_DEFAULTS["lr"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--local_rank", type=int, default=-1)
    args_main = parser.parse_args()

    if not args_main.run_all_experiments and not args_main.experiment:
        parser.error("Specify --experiment <name> or --run_all_experiments")

    try:
        import multiprocessing

        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    lmdb_paths = args_main.lmdb_paths or [DEFAULT_LMDB]

    if local_rank == 0:
        print(f"[NiComplex-OOD] LMDB={lmdb_paths}")

    all_valid, nan_skipped = load_merged_valid_nicomplex_records(lmdb_paths, local_rank=local_rank)
    if not all_valid:
        raise RuntimeError(f"Empty NiComplex LMDB: {lmdb_paths}")
    if nan_skipped > 0 and local_rank == 0:
        print(f"[NiComplex-OOD] skipped {nan_skipped} invalid ddG entries")
    if local_rank == 0:
        print(
            f"[NiComplex-OOD] loaded {len(all_valid)} valid samples | "
            f"R_Type={summarize_field(all_valid, 'R_Type')} | "
            f"L_Scaffold={summarize_field(all_valid, 'L_Scaffold')}"
        )

    experiments = list(EXPERIMENT_NAMES) if args_main.run_all_experiments else [args_main.experiment]
    for experiment_name in experiments:
        _train_one_experiment(args_main, experiment_name, all_valid, local_rank)
