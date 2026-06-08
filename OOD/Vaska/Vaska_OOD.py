"""
Vaska OOD training: leave-one-b_group-out (train on 4 groups, OOD test on the 5th).

Same model stack as Stage3/Vaska_Complex.py (single-token + SMILES + 3D).
Training hyperparameters follow train_defaults.VASKA_DEFAULTS (random-split baseline).

Example (one fold):
  deepspeed --num_gpus=2 OOD/Vaska/Vaska_OOD.py --holdout_b_group halides

Example (all 5 folds sequentially):
  deepspeed --num_gpus=2 OOD/Vaska/Vaska_OOD.py --run_all_loops
"""
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

from Vaska_Complex import VaskaComplexDataset, read_vaska_lmdb
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel
from train_defaults import VASKA_DEFAULTS

from OOD.Vaska.b_group_split import B_GROUPS, b_group_dirname, lobo_split_by_b_group, summarize_b_groups

DEFAULT_OUTPUT_DIR = "/data/jingyuan_data/Vaska_OOD_Models"


def _train_one_fold(args, holdout_b_group: str, all_raw: list, local_rank: int):
    train_samples, ood_test_samples = lobo_split_by_b_group(all_raw, holdout_b_group)
    fold_dir = os.path.join(args.output_dir, b_group_dirname(holdout_b_group))
    if local_rank == 0:
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, "ood_split.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "holdout_b_group": holdout_b_group,
                    "b_group_counts_all": summarize_b_groups(all_raw),
                    "n_train": len(train_samples),
                    "n_ood_test": len(ood_test_samples),
                },
                f,
                indent=2,
            )
        print(
            f"\n[Vaska-OOD] ===== holdout b_group={holdout_b_group} =====\n"
            f"  train={len(train_samples)} (all non-holdout) ood_test={len(ood_test_samples)}\n"
            f"  output_dir={fold_dir}"
        )
        wandb.init(
            project="Vaska_OOD_b_group",
            name=b_group_dirname(holdout_b_group),
            reinit=True,
            config={**vars(args), "holdout_b_group": holdout_b_group},
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

    train_dataset = VaskaComplexDataset(tokenizer=tokenizer, samples=train_samples)

    training_args = TrainingArguments(
        output_dir=fold_dir,
        num_train_epochs=args.epochs,
        deepspeed=os.path.join(_PROJECT_ROOT, "ds_config.json"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="no",
        report_to="wandb",
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
    trainer.save_model(fold_dir)

    if local_rank == 0:
        with open(os.path.join(fold_dir, "holdout_b_group.txt"), "w", encoding="utf-8") as f:
            f.write(holdout_b_group + "\n")
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vaska OOD: leave-one-b_group-out training")
    parser.add_argument("--model_name", type=str, default=VASKA_DEFAULTS["model_name"])
    parser.add_argument("--3D_encoder_dict", dest="three_d_encoder_dict", type=str, default=VASKA_DEFAULTS["3D_encoder_dict"])
    parser.add_argument("--Stage2_ckpt", dest="stage2_ckpt", type=str, default=VASKA_DEFAULTS["Stage2_ckpt"])
    parser.add_argument("--3D_encoder_ckpt", dest="three_d_encoder_ckpt", type=str, default=VASKA_DEFAULTS["3D_encoder_ckpt"])
    parser.add_argument("--lmdb", type=str, default=VASKA_DEFAULTS["lmdb"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--holdout_b_group", type=str, default=None, choices=list(B_GROUPS))
    parser.add_argument("--run_all_loops", action="store_true", help="Sequentially train all 5 LOBO folds")
    parser.add_argument("--lora_r", type=int, default=VASKA_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=VASKA_DEFAULTS["lora_alpha"])
    parser.add_argument("--lora_target", type=str, default=VASKA_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"])
    parser.add_argument("--projection_init", type=str, default=VASKA_DEFAULTS["projection_init"], choices=["pretrained", "from_scratch"])
    parser.add_argument("--epochs", type=int, default=VASKA_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=VASKA_DEFAULTS["lr"])
    parser.add_argument("--batch_size", type=int, default=VASKA_DEFAULTS["batch_size"])
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--local_rank", type=int, default=-1)
    args_main = parser.parse_args()

    if not args_main.run_all_loops and not args_main.holdout_b_group:
        parser.error("Specify --holdout_b_group <name> or --run_all_loops")

    try:
        import multiprocessing

        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(f"[Vaska-OOD] LMDB={args_main.lmdb} | b_groups={B_GROUPS}")

    all_raw = read_vaska_lmdb(args_main.lmdb, max_samples=None)
    if not all_raw:
        raise RuntimeError(f"Empty LMDB: {args_main.lmdb}")
    if local_rank == 0:
        print(f"[Vaska-OOD] loaded {len(all_raw)} samples | {summarize_b_groups(all_raw)}")

    folds = list(B_GROUPS) if args_main.run_all_loops else [args_main.holdout_b_group]
    for holdout in folds:
        _train_one_fold(args_main, holdout, all_raw, local_rank)
