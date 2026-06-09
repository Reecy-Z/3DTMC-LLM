"""
Unified Stage 3 training: TmQM properties, Vaska barrier, NiComplex ΔΔG.

Default (TmQM property from Stage2):
  deepspeed --num_gpus=2 Stage3.py --task homo_lumo_gap

Modes (--mode; ``--ablation`` is a deprecated alias):
  single_token  Default full SFT (instruction + SMILES + 3D)
  freeze_3d     Frozen 3D encoder + projection; train LoRA only
  random_3d     Random structure-slot embedding; train LoRA only
  multi_token   Learnable multi-token query projection
  3d_only       Instruction only (no SMILES in prompt)

Examples:
  deepspeed Stage3.py --task dipole_moment --mode freeze_3d --3D_encoder_ckpt /path/encoder.pt
  deepspeed Stage3.py --task vaska_barrier --lmdb /path/data.lmdb --split_seed 43
  deepspeed Stage3.py --task nicomplex_ddg --lmdb /path/a.lmdb --lmdb /path/b.lmdb
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import wandb
from transformers import AutoTokenizer, TrainingArguments

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import utils  # noqa: F401
from multimodal_LLM import (
    RECIPE_STAGE3,
    MultimodalModel,
    MultimodalModelFreeze3D,
    MultimodalModelMultiToken,
    MultimodalModelRandom3D,
    MultimodalMultiTokenTrainer,
)
from task_datasets import (
    NI_INSTRUCTION,
    PROPERTY_CONFIG,
    NiComplexDDGDataset,
    TmQMg3DOnlyUnimolDataset,
    TmQMgSingleTokenUnimolDataset,
    VaskaComplexDataset,
    load_merged_valid_nicomplex_records,
    read_vaska_lmdb,
)
from task_registry import (
    MODES_STAGE3,
    STAGE3_TASKS,
    get_task,
    normalize_task_name,
    resolve_instruction,
    wandb_project,
)
from train_defaults import (
    DESCRIPTION_DEFAULTS,
    NICOMPLEX_DEFAULTS,
    PROPERTY_DEFAULTS,
    VASKA_DEFAULTS,
)
from utils import (
    DESCRIPTION_SAVE_KWARGS,
    MAX_SEQ_LENGTH,
    DescriptionTrainer,
    MultimodalCollator,
    MultimodalFullTrainer,
    require_existing_model_path,
    resolve_three_d_encoder_ckpt,
)

TMQM_TASKS = frozenset({"dipole_moment", "polarisability", "homo_lumo_gap"})
RANDOM_SPLIT_TASKS = frozenset({"vaska_barrier", "nicomplex_ddg"})

ABLATION_TO_MODE = {
    "property": "single_token",
    "freeze_3d": "freeze_3d",
    "random_3d": "random_3d",
    "multi_token": "multi_token",
    "3d_only": "3d_only",
}

_WANDB_ABLATION_PROJECT = {
    "freeze_3d": "Property_ablation_freeze_3d",
    "random_3d": "Property_ablation_random_3d",
}


def _finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "property", None):
        args.task = args.property
    if getattr(args, "ablation", None) is not None:
        mapped = ABLATION_TO_MODE.get(args.ablation)
        if mapped is None:
            raise ValueError(f"Unknown --ablation={args.ablation!r}")
        if args.mode != "single_token" and args.mode != mapped:
            print(f"[Stage3] --ablation={args.ablation!r} overrides --mode -> {mapped!r}")
        args.mode = mapped
    args.task = normalize_task_name(args.task)
    if args.task not in STAGE3_TASKS:
        raise ValueError(f"Unknown --task={args.task!r}; choose from {list(STAGE3_TASKS)}")
    if args.mode not in MODES_STAGE3:
        raise ValueError(f"Unknown --mode={args.mode!r}; choose from {list(MODES_STAGE3)}")
    if args.task in RANDOM_SPLIT_TASKS and args.mode != "single_token":
        raise ValueError(f"Task {args.task!r} only supports --mode single_token")
    return args


def _task_defaults(task: str) -> dict:
    if task in TMQM_TASKS:
        return dict(PROPERTY_DEFAULTS)
    if task == "vaska_barrier":
        return dict(VASKA_DEFAULTS)
    if task == "nicomplex_ddg":
        return dict(NICOMPLEX_DEFAULTS)
    raise KeyError(task)


def _output_dir(args: argparse.Namespace, task: str) -> str:
    if args.output_dir:
        if task in RANDOM_SPLIT_TASKS:
            return os.path.join(args.output_dir, f"seed_{args.split_seed}")
        return args.output_dir
    spec = get_task(task)
    suffix = spec.output_dir_suffix or task
    if args.mode == "single_token":
        return f"/data/jingyuan_data/Stage3_{suffix}_ckpt"
    return f"/data/jingyuan_data/Stage3_{suffix}_{args.mode}_ckpt"


def _instruction_for_run(args: argparse.Namespace, task: str) -> str | None:
    if args.instruction:
        return args.instruction
    if task in RANDOM_SPLIT_TASKS:
        if task == "nicomplex_ddg":
            return NI_INSTRUCTION
        return resolve_instruction(task, mode="single_token")
    return resolve_instruction(
        task,
        mode=args.mode if args.mode != "3d_only" else "single_token",
        use_polished_description=args.use_polished_description,
    )


def _random_80_10_10(samples: list, split_seed: int):
    n_total = len(samples)
    indices = np.arange(n_total)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(indices)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]
    return train, val, test, n_total


def _effective_save_steps(args: argparse.Namespace, task: str) -> int:
    if args.save_steps != 1000:
        return args.save_steps
    if task == "vaska_barrier":
        return 50
    if task == "nicomplex_ddg":
        return 100
    return args.save_steps


def _build_training_args(
    args: argparse.Namespace,
    task: str,
    output_dir: str,
    *,
    eval_strategy="no",
    eval_steps=None,
    ddp_unused=False,
    save_kwargs=None,
    use_deepspeed: bool | None = None,
    grad_accum: int | None = None,
):
    ds_config = os.path.join(_PROJECT_ROOT, "ds_config.json")
    is_tmqm_default = task in TMQM_TASKS and args.mode == "single_token"
    if use_deepspeed is None:
        use_deepspeed = task == "vaska_barrier" or is_tmqm_default
    if grad_accum is None:
        grad_accum = 4 if is_tmqm_default else 1

    kw = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": 0.1 if (is_tmqm_default or task == "vaska_barrier") else 0.03,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01 if (is_tmqm_default or task == "vaska_barrier") else 0.05,
        "logging_steps": 10 if task != "nicomplex_ddg" else 100,
        "save_steps": _effective_save_steps(args, task),
        "save_strategy": "steps",
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "report_to": "wandb",
        "ddp_find_unused_parameters": ddp_unused,
        "label_names": ["labels"],
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 4,
        "dataloader_prefetch_factor": 2,
        "save_total_limit": 20 if (is_tmqm_default or task == "vaska_barrier") else 10,
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    if task in RANDOM_SPLIT_TASKS:
        kw["seed"] = args.split_seed
    if use_deepspeed:
        kw["deepspeed"] = ds_config
    if save_kwargs:
        kw.update(save_kwargs)
    return TrainingArguments(**kw)


def _build_single_token_model(args: argparse.Namespace, *, init_ckpt=None, load_pretrained_projection=True):
    return MultimodalModel(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=args.three_d_encoder_ckpt,
        init_ckpt=init_ckpt if init_ckpt is not None else args.stage2_ckpt,
        load_pretrained_projection=load_pretrained_projection,
        train_3d_encoder=True,
        train_projection=True,
        train_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
    )


def _run_tmqm_single_token(args: argparse.Namespace, local_rank: int, task: str, output_dir: str):
    instruction_use = _instruction_for_run(args, task)
    if local_rank == 0:
        print(
            f"[Stage3] task={task} mode=single_token; "
            f"prompt={'description+SMILES+3D' if args.use_polished_description else 'SMILES+3D'}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = _build_single_token_model(
        args,
        load_pretrained_projection=(args.projection_init == "pretrained"),
    )
    train_dataset = TmQMgSingleTokenUnimolDataset(
        [args.train_lmdb],
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        property_key=task,
        instruction=instruction_use,
        use_polished_description=args.use_polished_description,
    )
    val_dataset = TmQMgSingleTokenUnimolDataset(
        [args.val_lmdb],
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        property_key=task,
        instruction=instruction_use,
        use_polished_description=args.use_polished_description,
    )
    if local_rank == 0:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    training_args = _build_training_args(
        args,
        task,
        output_dir,
        eval_strategy="steps",
        eval_steps=10000,
        use_deepspeed=True,
        grad_accum=4,
    )
    trainer = MultimodalFullTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()


def _run_tmqm_freeze_or_random(args: argparse.Namespace, local_rank: int, task: str, output_dir: str):
    mode = args.mode
    instruction_use = resolve_instruction(task, mode="single_token")
    require_existing_model_path(args.model_name, "--model_name")
    encoder_ckpt = resolve_three_d_encoder_ckpt(args.three_d_encoder_ckpt, args.stage2_ckpt)

    if local_rank == 0:
        tag = "Freeze3D" if mode == "freeze_3d" else "Random3D"
        print(f"[Stage3-{tag}] task={task}; train LoRA only")
        print(f"[Stage3-{tag}] 3D encoder weights: {encoder_ckpt}")
        if mode == "random_3d":
            print(f"[Stage3-{tag}] random_3d_seed={args.random_3d_seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_kwargs = dict(
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=encoder_ckpt,
        init_ckpt=None,
        train_3d_encoder=False,
        train_projection=False,
        train_lora=True,
        load_pretrained_projection=False,
        load_pretrained_lora=False,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
    )
    if mode == "freeze_3d":
        model = MultimodalModelFreeze3D(args.model_name, args.three_d_encoder_dict, **model_kwargs)
    else:
        model = MultimodalModelRandom3D(
            args.model_name,
            args.three_d_encoder_dict,
            random_3d_seed=args.random_3d_seed,
            **model_kwargs,
        )
    if local_rank == 0:
        model.llm.print_trainable_parameters()

    train_dataset = TmQMgSingleTokenUnimolDataset(
        [args.train_lmdb],
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        property_key=task,
        instruction=instruction_use,
        use_polished_description=False,
    )
    if local_rank == 0:
        print(f"[Stage3-{mode}] Train samples: {len(train_dataset)} (no validation)")

    training_args = _build_training_args(
        args,
        task,
        output_dir,
        ddp_unused=(mode == "random_3d"),
        save_kwargs=DESCRIPTION_SAVE_KWARGS,
        use_deepspeed=True,
        grad_accum=4,
    )
    trainer = DescriptionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()


def _run_tmqm_multi_token(args: argparse.Namespace, local_rank: int, task: str, output_dir: str):
    instruction_use = resolve_instruction(task, mode="single_token")
    require_existing_model_path(args.model_name, "--model_name")
    encoder_ckpt = resolve_three_d_encoder_ckpt(args.three_d_encoder_ckpt, args.stage2_ckpt)
    init_ckpt = args.stage2_ckpt if args.lora_init == "pretrained" else None

    if local_rank == 0:
        print(
            f"[Stage3-multi_token] task={task}; lora_init={args.lora_init}; train_lora={args.train_lora}"
        )
        print(f"[Stage3-multi_token] 3D encoder weights: {encoder_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalModelMultiToken(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=encoder_ckpt,
        init_ckpt=init_ckpt,
        train_3d_encoder=args.train_3d_encoder,
        train_projection=args.train_projection,
        train_lora=args.train_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
        load_pretrained_lora=(args.lora_init == "pretrained"),
    )
    if local_rank == 0 and args.train_lora:
        model.llm.print_trainable_parameters()

    train_dataset = TmQMgSingleTokenUnimolDataset(
        [args.train_lmdb],
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        property_key=task,
        instruction=instruction_use,
        use_polished_description=False,
    )
    if local_rank == 0:
        print(f"[Stage3-multi_token] Train samples: {len(train_dataset)} (no validation)")

    training_args = _build_training_args(
        args,
        task,
        output_dir,
        save_kwargs=DESCRIPTION_SAVE_KWARGS,
        use_deepspeed=True,
        grad_accum=4,
    )
    trainer = MultimodalMultiTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()


def _run_tmqm_3d_only(args: argparse.Namespace, local_rank: int, task: str, output_dir: str):
    require_existing_model_path(args.model_name, "--model_name")
    encoder_ckpt = resolve_three_d_encoder_ckpt(args.three_d_encoder_ckpt, args.stage2_ckpt)
    init_ckpt = args.stage2_ckpt if args.lora_init == "pretrained" else None

    if local_rank == 0:
        print(f"[Stage3-3d_only] task={task}; prompt=instruction only (NO SMILES)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultimodalModel(
        args.model_name,
        args.three_d_encoder_dict,
        recipe=RECIPE_STAGE3,
        three_d_encoder_ckpt=encoder_ckpt,
        init_ckpt=init_ckpt,
        train_3d_encoder=args.train_3d_encoder,
        train_projection=args.train_projection,
        train_lora=args.train_lora,
        load_pretrained_projection=False,
        load_pretrained_lora=(args.lora_init == "pretrained"),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
    )
    if local_rank == 0 and args.train_lora:
        model.llm.print_trainable_parameters()

    train_dataset = TmQMg3DOnlyUnimolDataset(
        [args.train_lmdb],
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        property_key=task,
    )
    if local_rank == 0:
        print(f"[Stage3-3d_only] Train samples: {len(train_dataset)} (no validation)")

    training_args = _build_training_args(
        args,
        task,
        output_dir,
        save_kwargs=DESCRIPTION_SAVE_KWARGS,
        use_deepspeed=True,
        grad_accum=4,
    )
    trainer = MultimodalFullTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()


def _run_vaska(args: argparse.Namespace, local_rank: int, output_dir: str):
    split_seed = args.split_seed
    if local_rank == 0:
        print(f"[Stage3] task=vaska_barrier mode=single_token | split_seed={split_seed}")

    lmdb_path = (args.lmdb_paths or [VASKA_DEFAULTS["lmdb"]])[0]
    all_raw = read_vaska_lmdb(lmdb_path, max_samples=None)
    if not all_raw:
        raise RuntimeError(f"[Stage3] LMDB {lmdb_path} has no samples")

    train_samples, val_samples, test_samples, n_total = _random_80_10_10(all_raw, split_seed)
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(
            f"[Stage3] vaska_barrier | total={n_total}, train={len(train_samples)}, "
            f"val={len(val_samples)}, test={len(test_samples)} | output_dir={output_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = _build_single_token_model(
        args,
        load_pretrained_projection=(args.projection_init == "pretrained"),
    )
    train_dataset = VaskaComplexDataset(tokenizer=tokenizer, samples=train_samples)
    val_dataset = VaskaComplexDataset(tokenizer=tokenizer, samples=val_samples)

    training_args = _build_training_args(
        args,
        "vaska_barrier",
        output_dir,
        eval_strategy="steps",
        eval_steps=1000,
        use_deepspeed=True,
        grad_accum=1,
    )
    trainer = MultimodalFullTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(output_dir)
    if local_rank == 0:
        with open(os.path.join(output_dir, "split_seed.txt"), "w", encoding="utf-8") as f:
            f.write(f"{split_seed}\n")


def _run_nicomplex(args: argparse.Namespace, local_rank: int, output_dir: str):
    split_seed = args.split_seed
    lmdb_paths = args.lmdb_paths or NICOMPLEX_DEFAULTS["lmdb"]

    merged_valid, nan_skipped = load_merged_valid_nicomplex_records(lmdb_paths, local_rank=local_rank)
    if not merged_valid:
        raise RuntimeError(f"[Stage3] No valid ddG samples: {lmdb_paths}")
    if nan_skipped > 0 and local_rank == 0:
        print(f"[Stage3] Skipped {nan_skipped} invalid ddG entries while merging")

    train_samples, val_samples, test_samples, n_total = _random_80_10_10(merged_valid, split_seed)
    if args.max_train_samples is not None:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_eval_samples is not None:
        val_samples = val_samples[: args.max_eval_samples]

    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(
            f"[Stage3] nicomplex_ddg | split_seed={split_seed} | total={n_total}, "
            f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)} | "
            f"output_dir={output_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = _build_single_token_model(
        args,
        load_pretrained_projection=(args.projection_init == "pretrained"),
    )
    instruction_use = _instruction_for_run(args, "nicomplex_ddg")
    train_dataset = NiComplexDDGDataset(
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
        samples=train_samples,
        instruction=instruction_use,
    )
    eval_dataset = NiComplexDDGDataset(
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
        samples=val_samples,
        instruction=instruction_use,
    )

    training_args = _build_training_args(
        args,
        "nicomplex_ddg",
        output_dir,
        eval_strategy="steps",
        eval_steps=2000,
        use_deepspeed=True,
        grad_accum=2 if args.batch_size >= 64 else 1,
    )
    trainer = MultimodalFullTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MultimodalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(output_dir)
    if local_rank == 0:
        with open(os.path.join(output_dir, "split_seed.txt"), "w", encoding="utf-8") as f:
            f.write(f"{split_seed}\n")


def _init_wandb(args: argparse.Namespace, task: str, local_rank: int) -> None:
    if local_rank != 0:
        return
    if args.mode in _WANDB_ABLATION_PROJECT:
        wandb.init(project=_WANDB_ABLATION_PROJECT[args.mode])
        return
    if task in RANDOM_SPLIT_TASKS:
        spec = get_task(task)
        wandb.init(
            project=spec.wandb_project_prefix or f"Stage3_{task}",
            name=f"seed_{args.split_seed}",
            reinit=True,
            config=vars(args),
        )
        return
    wandb.init(project=wandb_project(task, args.mode))


def run_training(args: argparse.Namespace) -> None:
    try:
        import multiprocessing

        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    args = _finalize_args(args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    task = args.task
    output_dir = _output_dir(args, task)

    _init_wandb(args, task, local_rank)

    if task in TMQM_TASKS:
        if args.mode == "single_token":
            _run_tmqm_single_token(args, local_rank, task, output_dir)
        elif args.mode in ("freeze_3d", "random_3d"):
            _run_tmqm_freeze_or_random(args, local_rank, task, output_dir)
        elif args.mode == "multi_token":
            _run_tmqm_multi_token(args, local_rank, task, output_dir)
        elif args.mode == "3d_only":
            _run_tmqm_3d_only(args, local_rank, task, output_dir)
        else:
            raise ValueError(f"Unsupported mode={args.mode!r} for task={task!r}")
    elif task == "vaska_barrier":
        _run_vaska(args, local_rank, output_dir)
    elif task == "nicomplex_ddg":
        _run_nicomplex(args, local_rank, output_dir)
    else:
        raise ValueError(f"Unhandled task={task!r}")

    if local_rank == 0:
        wandb.finish()


def build_parser(*, default_task: str | None = None) -> argparse.ArgumentParser:
    default_task = default_task or PROPERTY_DEFAULTS["property"]
    defaults = _task_defaults(default_task)

    parser = argparse.ArgumentParser(description="Stage 3: unified regression / downstream training")
    parser.add_argument(
        "--task",
        type=str,
        default=default_task,
        choices=list(STAGE3_TASKS),
        help="Training task (TmQM property, vaska_barrier, nicomplex_ddg)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single_token",
        choices=list(MODES_STAGE3),
        help="Model/input mode (replaces legacy --ablation)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=list(ABLATION_TO_MODE.keys()),
        help="Deprecated alias for --mode (property -> single_token)",
    )
    parser.add_argument("--model_name", type=str, default=defaults.get("model_name", PROPERTY_DEFAULTS["model_name"]))
    parser.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=defaults.get("3D_encoder_dict", PROPERTY_DEFAULTS["3D_encoder_dict"]),
    )
    parser.add_argument(
        "--Stage2_ckpt",
        dest="stage2_ckpt",
        type=str,
        default=defaults.get("Stage2_ckpt", PROPERTY_DEFAULTS["Stage2_ckpt"]),
    )
    parser.add_argument(
        "--3D_encoder_ckpt",
        dest="three_d_encoder_ckpt",
        type=str,
        default=defaults.get("3D_encoder_ckpt", PROPERTY_DEFAULTS["3D_encoder_ckpt"]),
    )
    parser.add_argument("--train_lmdb", type=str, default=PROPERTY_DEFAULTS["train_lmdb"])
    parser.add_argument("--val_lmdb", type=str, default=PROPERTY_DEFAULTS["val_lmdb"])
    parser.add_argument(
        "--lmdb",
        action="append",
        default=None,
        dest="lmdb_paths",
        metavar="PATH",
        help="LMDB for vaska_barrier (once) or nicomplex_ddg (repeatable)",
    )
    parser.add_argument("--output_dir", type=str, default=defaults.get("output_dir", PROPERTY_DEFAULTS["output_dir"]))
    parser.add_argument(
        "--property",
        type=str,
        default=None,
        choices=list(TMQM_TASKS),
        help="Deprecated alias for --task (TmQM properties only)",
    )
    parser.add_argument("--lora_r", type=int, default=defaults.get("lora_r", PROPERTY_DEFAULTS["lora_r"]))
    parser.add_argument("--lora_alpha", type=int, default=defaults.get("lora_alpha", PROPERTY_DEFAULTS["lora_alpha"]))
    parser.add_argument(
        "--lora_target",
        type=str,
        default=defaults.get("lora_target", PROPERTY_DEFAULTS["lora_target"]),
        choices=["qv", "qkv", "all"],
    )
    parser.add_argument(
        "--projection_init",
        type=str,
        default=defaults.get("projection_init", PROPERTY_DEFAULTS["projection_init"]),
        choices=["pretrained", "from_scratch"],
    )
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", PROPERTY_DEFAULTS["epochs"]))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", PROPERTY_DEFAULTS["lr"]))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", PROPERTY_DEFAULTS["batch_size"]))
    parser.add_argument(
        "--use_polished_description",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="TmQM homo_lumo_gap: prepend polished_description in prompt",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Override task instruction from task_registry",
    )
    parser.add_argument("--split_seed", type=int, default=defaults.get("split_seed", VASKA_DEFAULTS["split_seed"]))
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--random_3d_seed", type=int, default=DESCRIPTION_DEFAULTS["random_3d_seed"])
    parser.add_argument(
        "--lora_init",
        type=str,
        default="from_scratch",
        choices=["from_scratch", "pretrained"],
    )
    parser.add_argument("--train_3d_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser


def main(argv: list[str] | None = None, *, default_task: str | None = None) -> None:
    parser = build_parser(default_task=default_task)
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
