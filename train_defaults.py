"""Centralized argparse defaults for 3DTMC training scripts."""

STAGE1_DEFAULTS = {
    "model_name": "/path/to/HF_models/Qwen3-4B-Instruct-2507",
    "unimol_ckpt": "/path/to/STAGE1/checkpoint",
    "unimol_dict": "/path/to/dict.txt",
    "train_lmdb": [
        "/path/to/train/tmc.lmdb",
    ],
    "val_lmdb": "/path/to/valid/tmc.lmdb",
    "output_dir": "/path/to/STAGE1",
}

STAGE2_DEFAULTS = {
    "model_name": "/path/to/HF_models/Qwen3-4B-Instruct-2507",
    "adapter": "/path/to/STAGE1/checkpoint",
    "unimol_dict": "/path/to/dict.txt",
    "train_lmdb": [
        "/path/to/train/tmc.lmdb",
    ],
    "val_lmdb": "/path/to/valid/tmc.lmdb",
    "json_qa": [
        "/path/to/smiles_qa_output.json",
        "/path/to/coordination_chemistry_qa_output.json",
        "/path/to/tmc_basic_qa_output.json",
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
    "unimol_ckpt": None,
    "unimol_dict": "/path/to/dict.txt",
    "init_ckpt": "/path/to/STAGE2/checkpoint",
    "train_lmdb": "/path/to/train/data.lmdb",
    "val_lmdb": "/path/to/valid/data.lmdb",
    "output_dir": "/path/to/PROPERTY",
    "property": "homo_lumo_gap",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "epochs": 3,
    "lr": 3e-5,
    "batch_size": 16,
}

NICOMPLEX_DEFAULTS = {
    "model_name": "/path/to/HF_models/Qwen3-4B-Instruct-2507",
    "unimol_ckpt": None,
    "unimol_dict": "/path/to/dict.txt",
    "init_ckpt": "/path/to/STAGE2/checkpoint",
    "output_dir": "/path/to/NICOMPLEX",
    "lmdb": ["/path/to/data.lmdb"],
    "split_seed": 38,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "epochs": 20,
    "lr": 3e-5,
    "batch_size": 32,
}

VASKA_DEFAULTS = {
    "model_name": "/path/to/HF_models/Qwen3-4B-Instruct-2507",
    "unimol_ckpt": None,
    "unimol_dict": "/path/to/dict.txt",
    "init_ckpt": "/path/to/STAGE2/checkpoint",
    "lmdb": "/path/to/data.lmdb",
    "output_dir": "/path/to/VASKA",
    "split_seed": 43,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "epochs": 20,
    "lr": 3e-5,
    "batch_size": 32,
}
