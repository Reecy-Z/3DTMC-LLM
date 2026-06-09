"""Centralized argparse defaults for 3DTMC training scripts."""

# Uni-Core source root used by local imports (e.g., unicore.*).
from pickle import NONE


UNICORE_ROOT = "/home/featurize/work/Uni-Core"

STAGE1_DEFAULTS = {
    "model_name": "/path/to/HF_models/Qwen3-4B-Instruct-2507",
    "3D_encoder_ckpt": "/path/to/STAGE1/checkpoint",
    "3D_encoder_dict": "/path/to/3D_encoder_dict.txt",
    "train_lmdb": [
        "/path/to/train/tmc.lmdb",
    ],
    "val_lmdb": "/path/to/valid/tmc.lmdb",
    "output_dir": "/path/to/STAGE1",
    "epochs": 3,
    "lr": 2e-4,
    "batch_size": 4,
    "save_steps": 1,
}

STAGE2_DEFAULTS = {
    "model_name": "/path/to/HF_models/Qwen3-4B-Instruct-2507",
    "Stage1_ckpt": "/path/to/STAGE1/checkpoint",
    "3D_encoder_dict": "/path/to/3D_encoder_dict.txt",
    "train_lmdb": [
        "/path/to/train/tmc.lmdb",
    ],
    "val_lmdb": "/path/to/valid/tmc.lmdb",
    "json_qa": [
        "/path/to/coordination_chemistry_qa.json",
        "/path/to/organometallic_chemistry_qa.json",
    ],
    "output_dir": "/path/to/STAGE2",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "epochs": 3,
    "lr": 3e-5,
    "batch_size": 4,
}

PROPERTY_DEFAULTS = {
    "model_name": "/path/to/HF_models/Qwen3-4B-Instruct-2507",
    "3D_encoder_dict": "/path/to/3D_encoder_dict.txt",
    "Stage2_ckpt": "/path/to/STAGE2/checkpoint",
    "3D_encoder_ckpt": None,
    "train_lmdb": "/path/to/train/data.lmdb",
    "val_lmdb": "/path/to/valid/data.lmdb",
    "output_dir": "/path/to/PROPERTY",
    "property": "homo_lumo_gap",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "projection_init": "pretrained",
    "epochs": 3,
    "lr": 3e-5,
    "batch_size": 16,
}

NICOMPLEX_DEFAULTS = {
    "model_name": "/path/to/HF_models/Qwen3-4B-Instruct-2507",
    "3D_encoder_dict": "/path/to/3D_encoder_dict.txt",
    "Stage2_ckpt": "/path/to/STAGE2/checkpoint",
    "3D_encoder_ckpt": None,
    "output_dir": "/path/to/NICOMPLEX",
    "lmdb": ["/path/to/data.lmdb"],
    "split_seed": 38,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "projection_init": "pretrained",
    "epochs": 20,
    "lr": 3e-5,
    "batch_size": 32,
}

VASKA_DEFAULTS = {
    "model_name": "/path/to/HF_models/Qwen3-4B-Instruct-2507",
    "3D_encoder_dict": "/path/to/3D_encoder_dict.txt",
    "Stage2_ckpt": "/path/to/Stage2",
    "3D_encoder_ckpt": None,
    "lmdb": "/path/to/vaskas-space/data.lmdb",
    "output_dir": "VASKA",
    "split_seed": 43,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "projection_init": "pretrained",
    "epochs": 20,
    "lr": 3e-5,
    "batch_size": 32,
}

DESCRIPTION_DEFAULTS = {
    **STAGE2_DEFAULTS,
    "model_name": VASKA_DEFAULTS["model_name"],
    "3D_encoder_dict": VASKA_DEFAULTS["3D_encoder_dict"],
    "random_3d_seed": 42,
    "test_lmdb": "/home/featurize/TMC-Prop3D-Enriched/test/tmc.lmdb",
}