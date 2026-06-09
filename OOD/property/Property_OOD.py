"""
OOD Property training: standard Stage3 single-token + Qwen3 flow,
with train/test split from cluster_split_k150_far_from_train.csv (cluster OOD).

LMDB keys are CSD codes matching the CSV ``CSD code`` column.

Example:
  deepspeed --num_gpus=2 OOD/property/Property_OOD.py --property dipole_moment
"""
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

from Property import PROPERTY_CONFIG
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel
from train_defaults import PROPERTY_DEFAULTS, VASKA_DEFAULTS

from OOD.property.cluster_split import split_output_suffix
from OOD.property.dataset_ood import TmQMgClusterSplitDataset

DEFAULT_STAGE2_CKPT = "/data/jingyuan_data/3DTMC-LLM/Stage2"
# Override with --split_csv pointing to your train/test CSV.
DEFAULT_SPLIT_CSV = "/path/to/your_property_split.csv"
DEFAULT_EPOCHS = 10
DEFAULT_SAVE_STEPS = 1000
DEFAULT_LMDB_PATHS = [
    "/data/jingyuan_data/tmqmg/stage3/train/tmqmg_atom_only_new.lmdb",
    "/data/jingyuan_data/tmqmg/stage3/test/tmqmg_atom_only_new.lmdb",
]

PROPERTY_OOD_DEFAULTS = {
    **PROPERTY_DEFAULTS,
    "model_name": VASKA_DEFAULTS["model_name"],
    "3D_encoder_dict": VASKA_DEFAULTS["3D_encoder_dict"],
    "Stage2_ckpt": DEFAULT_STAGE2_CKPT,
    "property": "dipole_moment",
    "epochs": DEFAULT_EPOCHS,
    "save_steps": DEFAULT_SAVE_STEPS,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OOD Property: cluster k150 far-from-train split + single-token 3D + Qwen3 SFT"
    )
    parser.add_argument("--model_name", type=str, default=PROPERTY_OOD_DEFAULTS["model_name"])
    parser.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=PROPERTY_OOD_DEFAULTS["3D_encoder_dict"],
    )
    parser.add_argument("--Stage2_ckpt", dest="stage2_ckpt", type=str, default=PROPERTY_OOD_DEFAULTS["Stage2_ckpt"])
    parser.add_argument(
        "--3D_encoder_ckpt",
        dest="three_d_encoder_ckpt",
        type=str,
        default=PROPERTY_DEFAULTS["3D_encoder_ckpt"],
    )
    parser.add_argument(
        "--lmdb_paths",
        nargs="+",
        default=DEFAULT_LMDB_PATHS,
        help="One or more LMDB files (keys = CSD codes); filtered by split CSV or split directory.",
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default=DEFAULT_SPLIT_CSV,
        help="Combined split CSV (CSD+split columns), or directory with train.csv and test.csv.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--property",
        type=str,
        default=PROPERTY_OOD_DEFAULTS["property"],
        choices=["dipole_moment", "polarisability", "homo_lumo_gap"],
    )
    parser.add_argument("--lora_r", type=int, default=PROPERTY_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=PROPERTY_DEFAULTS["lora_alpha"])
    parser.add_argument("--lora_target", type=str, default=PROPERTY_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"])
    parser.add_argument(
        "--projection_init",
        type=str,
        default=PROPERTY_DEFAULTS["projection_init"],
        choices=["pretrained", "from_scratch"],
    )
    parser.add_argument(
        "--use_polished_description",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--epochs", type=int, default=PROPERTY_OOD_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=PROPERTY_DEFAULTS["lr"])
    parser.add_argument("--batch_size", type=int, default=PROPERTY_DEFAULTS["batch_size"])
    parser.add_argument("--save_steps", type=int, default=PROPERTY_OOD_DEFAULTS["save_steps"])
    parser.add_argument("--eval_steps", type=int, default=1000)
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

    if args_main.use_polished_description:
        instruction_use = prop_cfg.get("instruction_description") or prop_cfg["instruction_smiles"]
    else:
        instruction_use = prop_cfg.get("instruction_smiles") or prop_cfg.get("instruction_description")

    split_tag = split_output_suffix(args_main.split_csv)
    if local_rank == 0:
        wandb.init(project=f"OOD_Property_{prop_cfg['output_dir_suffix']}_{split_tag}")
        print(
            f"[Property-OOD] property={args_main.property}; split={args_main.split_csv}; "
            f"tag={split_tag}; "
            f"prompt={'description+SMILES+3D' if args_main.use_polished_description else 'SMILES+3D'}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args_main.model_name)
    model = MultimodalModel(
        args_main.model_name,
        args_main.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=args_main.three_d_encoder_ckpt,
        init_ckpt=args_main.stage2_ckpt,
        load_pretrained_projection=(args_main.projection_init == "pretrained"),
        train_3d_encoder=True,
        train_projection=True,
        train_lora=True,
        lora_r=args_main.lora_r,
        lora_alpha=args_main.lora_alpha,
        lora_target=args_main.lora_target,
    )

    ds_common = dict(
        lmdb_paths=args_main.lmdb_paths,
        split_csv=args_main.split_csv,
        tokenizer=tokenizer,
        property_key=prop_key,
        instruction=instruction_use,
        use_polished_description=args_main.use_polished_description,
    )
    train_dataset = TmQMgClusterSplitDataset(**ds_common, split_name="train")
    eval_dataset = TmQMgClusterSplitDataset(**ds_common, split_name="test")

    if local_rank == 0:
        print(f"[Property-OOD] Train (cluster train): {len(train_dataset)} | Eval (cluster test): {len(eval_dataset)}")

    output_dir = args_main.output_dir or (
        f"/data/jingyuan_data/OOD_Property_{prop_cfg['output_dir_suffix']}_{split_tag}_ckpt"
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
        eval_strategy="steps",
        eval_steps=args_main.eval_steps,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        save_total_limit=20,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = MultimodalFullTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()
