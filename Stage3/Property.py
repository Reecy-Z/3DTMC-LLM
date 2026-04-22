"""
3D encoder global representation (single-token / [CLS]) is projected by the single-token projection layer to Qwen3 hidden size and
concatenated with text at the object_ref slot.

Generative training only (CE). Can resume from checkpoints containing 3D_encoder.pt, single_token_projection.pt, and LoRA.

Examples:
  CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 Property.py
  CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 Property.py --Stage2_ckpt /path/to/checkpoint-xxx
"""
import os
import sys
import json
import pickle
import lmdb
import numpy as np
import wandb
from transformers import AutoTokenizer, TrainingArguments
from torch.utils.data import Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import utils  # noqa: F401 -- 3D encoder (unimol) sys.path
from utils import (
    MultimodalCollator,
    MultimodalFullTrainer,
    _atoms_coords_remove_h_center,
    _lmdb_env_kwargs,
    format_instruction_field,
    tokenize_generation_sample_object_ref,
)
from multimodal_LLM import RECIPE_STAGE3, MultimodalModel
from train_defaults import PROPERTY_DEFAULTS

# Property task definitions
PROPERTY_CONFIG = {
    "dipole_moment": {
        "key": "dipole_moment",
        "unit": "Debye",
        "instruction_smiles": "What is the dipole moment (in Debye) of this transition metal complex? Given the SMILES and structure, respond with the numerical value only:",
        "output_dir_suffix": "dipole_moment",
    },
    "polarisability": {
        "key": "polarisability",
        "unit": "Bohr^3",
        "instruction_smiles": "What is the polarisability (in Bohr^3) of this transition metal complex? Given the SMILES and structure, respond with the numerical value only:",
        "output_dir_suffix": "polarisability",
    },
    "homo_lumo_gap": {
        "key": "homo_lumo_gap",
        "unit": "Ha",
        "instruction_description": "What is the HOMO-LUMO gap (in Ha) of this transition metal complex? Given the description, SMILES and structure, respond with the numerical value only:",
        "output_dir_suffix": "homo_lumo_gap_description",
    },
}


class TmQMgSingleTokenUnimolDataset(Dataset):
    def __init__(
        self,
        lmdb_paths,
        tokenizer,
        max_samples=None,
        property_key="dipole_moment",
        instruction=None,
    ):
        self.tokenizer = tokenizer
        self.property_key = property_key
        pcfg = PROPERTY_CONFIG.get(property_key, PROPERTY_CONFIG["dipole_moment"])
        self.instruction = instruction or pcfg.get("instruction_description") or pcfg["instruction_smiles"]
        self.key_index = []
        self._envs = {}
        nan_skipped = 0
        smiles_skipped = 0
        stop = False
        for p in lmdb_paths:
            if not os.path.exists(p):
                continue
            env = lmdb.open(p, **_lmdb_env_kwargs())
            try:
                with env.begin() as txn:
                    cursor = txn.cursor()
                    for key_bytes, value in cursor:
                        d = pickle.loads(value)
                        if "atoms" not in d or "coordinates" not in d:
                            continue
                        if "smiles" not in d:
                            continue
                        if not format_instruction_field(d.get("smiles")):
                            smiles_skipped += 1
                            continue
                        if property_key not in d:
                            continue
                        try:
                            val = float(d[property_key])
                            if np.isnan(val):
                                nan_skipped += 1
                                continue
                        except (ValueError, TypeError):
                            nan_skipped += 1
                            continue
                        self.key_index.append((p, key_bytes))
                        if max_samples is not None and len(self.key_index) >= max_samples:
                            stop = True
                            break
                        d = None
                    if stop:
                        break
            finally:
                env.close()
            if stop:
                break
        if not self.key_index and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            raise RuntimeError(f"No valid samples found: {lmdb_paths}")
        if nan_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Dataset] Skipped {nan_skipped} samples with NaN {property_key}")
        if smiles_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Dataset] Skipped {smiles_skipped} samples (missing or empty SMILES)")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Dataset] property={property_key}, {len(self.key_index)} samples; mode=GENERATION (with SMILES)")

    def __len__(self):
        return len(self.key_index)

    def __getitem__(self, idx):
        lmdb_path, key_bytes = self.key_index[idx]
        env = self._envs.get(lmdb_path)
        if env is None:
            env = lmdb.open(lmdb_path, **_lmdb_env_kwargs())
            self._envs[lmdb_path] = env
        with env.begin() as txn:
            raw = txn.get(key_bytes)
            if raw is None:
                raise KeyError(f"LMDB key not found: {key_bytes!r}")
            data = pickle.loads(raw)
        smiles = format_instruction_field(data.get("smiles"))
        response = str(data[self.property_key])
        atoms = data["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(data["coordinates"], dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
        if self.property_key == "homo_lumo_gap" and "polished_description" in data:
            desc = data.get("polished_description")
            if isinstance(desc, bytes):
                desc = desc.decode("utf-8", errors="ignore")
            if desc is not None:
                desc = str(desc).strip()
            if desc:
                user_content = f"{desc}\n{self.instruction} {smiles}"
            else:
                user_content = f"{self.instruction} {smiles}"
        else:
            user_content = f"{self.instruction} {smiles}"
        ids = tokenize_generation_sample_object_ref(self.tokenizer, user_content, response)
        return {**ids, "atoms": atoms, "coordinates": coords}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Property: 3D encoder single-token + Qwen3 generative SFT")
    parser.add_argument(
        "--model_name",
        type=str,
        default=PROPERTY_DEFAULTS["model_name"],
    )
    parser.add_argument(
        "--3D_encoder_dict",
        dest="three_d_encoder_dict",
        type=str,
        default=PROPERTY_DEFAULTS["3D_encoder_dict"],
    )
    parser.add_argument(
        "--Stage2_ckpt",
        dest="stage2_ckpt",
        type=str,
        default=PROPERTY_DEFAULTS["Stage2_ckpt"],
        help="HF-style checkpoint dir: adapter, 3D_encoder.pt, single_token_projection.pt (optional)",
    )
    parser.add_argument(
        "--3D_encoder_ckpt",
        dest="three_d_encoder_ckpt",
        type=str,
        default=PROPERTY_DEFAULTS["3D_encoder_ckpt"],
        help="Optional override for 3D encoder ckpt. If set, it takes priority over Stage2_ckpt/3D_encoder.pt.",
    )
    parser.add_argument(
        "--train_lmdb",
        type=str,
        default=PROPERTY_DEFAULTS["train_lmdb"],
    )
    parser.add_argument(
        "--val_lmdb",
        type=str,
        default=PROPERTY_DEFAULTS["val_lmdb"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=PROPERTY_DEFAULTS["output_dir"],
        help="Training output dir; default /data/jingyuan_data/Stage3_Property_<property>_multimodal_ckpt",
    )
    parser.add_argument("--property", type=str, default=PROPERTY_DEFAULTS["property"],
                        choices=["dipole_moment", "polarisability", "homo_lumo_gap"],
                        help="Target property to train")
    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=PROPERTY_DEFAULTS["lora_r"], help="LoRA rank (default 8; higher rank = more capacity)")
    parser.add_argument("--lora_alpha", type=int, default=PROPERTY_DEFAULTS["lora_alpha"], help="LoRA alpha (default 32)")
    parser.add_argument("--lora_target", type=str, default=PROPERTY_DEFAULTS["lora_target"], choices=["qv", "qkv", "all"],
                        help="LoRA target modules: qv=[q,v]_proj, qkv=[q,k,v]_proj, all=[q,k,v,o,gate,up,down]_proj")
    parser.add_argument(
        "--projection_init",
        type=str,
        default=PROPERTY_DEFAULTS["projection_init"],
        choices=["pretrained", "from_scratch"],
        help="Initialize projection from Stage2 ckpt or from scratch.",
    )
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=PROPERTY_DEFAULTS["epochs"], help="Number of epochs (default 3)")
    parser.add_argument("--lr", type=float, default=PROPERTY_DEFAULTS["lr"], help="Learning rate (default 5e-5)")
    parser.add_argument("--batch_size", type=int, default=PROPERTY_DEFAULTS["batch_size"], help="per_device_train_batch_size (default 4)")
    parser.add_argument("--local_rank", type=int, default=-1)
    args_main = parser.parse_args()
    try:
        import multiprocessing
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Selected property config
    prop_cfg = PROPERTY_CONFIG[args_main.property]
    prop_key = prop_cfg["key"]

    instruction_use = prop_cfg.get("instruction_description") or prop_cfg["instruction_smiles"]

    if local_rank == 0:
        wandb.init(project=f"Stage3_Property_{prop_cfg['output_dir_suffix']}")
        print(f"[Property] property={args_main.property}, generation (single-token projection + CE)")

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
    train_dataset = TmQMgSingleTokenUnimolDataset(
        [args_main.train_lmdb],
        tokenizer=tokenizer,
        max_samples=None,
        property_key=prop_key,
        instruction=instruction_use,
    )
    val_dataset = TmQMgSingleTokenUnimolDataset(
        [args_main.val_lmdb],
        tokenizer=tokenizer,
        max_samples=None,
        property_key=prop_key,
        instruction=instruction_use,
    )
    if local_rank == 0:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    data_collator = MultimodalCollator(tokenizer)
    ds_config = os.path.join(_PROJECT_ROOT, "ds_config.json")
    label_names = ["labels"]
    output_dir = args_main.output_dir or f"/data/jingyuan_data/Stage3_Property_{prop_cfg['output_dir_suffix']}_ckpt"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args_main.epochs,
        deepspeed=ds_config,
        per_device_train_batch_size=args_main.batch_size,
        per_device_eval_batch_size=args_main.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args_main.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        save_steps=1000,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=10000,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        label_names=label_names,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        save_total_limit=20,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = MultimodalFullTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)
    trainer.train()
