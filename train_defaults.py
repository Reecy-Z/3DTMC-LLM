"""Centralized argparse defaults for 3DTMC training scripts."""

STAGE1_DEFAULTS = {
    "model_name": "/data/jingyuan_data/HF_models/Qwen3-4B-Instruct-2507",
    "unimol_ckpt": "/data/jingyuan_data/checkpoint-unimol-OMol-all-500000",
    "unimol_dict": "/data/jingyuan_data/OMol25_MC_all/dict.txt",
    "train_lmdb": [
        "/data/jingyuan_data/tmQM/stage1/train/tmc.lmdb",
        "/data/jingyuan_data/tmQM/stage1/valid/tmc.lmdb",
    ],
    "val_lmdb": "/data/jingyuan_data/tmQM/stage1/valid/tmc.lmdb",
    "output_dir": "/data/jingyuan_data/sft_intro_bridge_unimol_ckpt_bos_only",
}

STAGE2_DEFAULTS = {
    "model_name": "/data/jingyuan_data/HF_models/Qwen3-4B-Instruct-2507",
    "adapter": "/data/jingyuan_data/sft_intro_bridge_unimol_ckpt_bos_only/checkpoint-7821",
    "unimol_dict": "/data/jingyuan_data/OMol25_MC_all/dict.txt",
    "train_lmdb": [
        "/data/jingyuan_data/tmQM/stage1/train/tmc.lmdb",
        "/data/jingyuan_data/tmQM/stage1/valid/tmc.lmdb",
    ],
    "val_lmdb": "/data/jingyuan_data/tmQM/stage1/test/tmc.lmdb",
    "json_qa": [
        "/data/jingyuan_data/TMC_JSON/smiles_qa_output.json",
        "/data/jingyuan_data/TMC_JSON/coordination_chemistry_qa_output.json",
        "/data/jingyuan_data/TMC_JSON/tmc_basic_qa_output.json",
    ],
    "output_dir": "/data/jingyuan_data/sft_intro_bridge_unimol_continue_ckpt",
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_target": "qv",
    "epochs": 3,
    "lr": 2e-4,
    "batch_size": 4,
}

PROPERTY_DEFAULTS = {
    "model_name": "/data/jingyuan_data/HF_models/Qwen3-4B-Instruct-2507",
    "unimol_ckpt": None,
    "unimol_dict": "/data/jingyuan_data/OMol25_MC_all/dict.txt",
    "init_ckpt": "/data/jingyuan_data/sft_intro_bridge_unimol_continue_ckpt_bos_only/checkpoint-7956",
    "train_lmdb": "/data/jingyuan_data/tmqmg/stage3/train/tmqmg_atom_only_new.lmdb",
    "val_lmdb": "/data/jingyuan_data/tmqmg/stage3/test/tmqmg_atom_only_new.lmdb",
    "output_dir": None,
    "property": "homo_lumo_gap",
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_target": "qv",
    "epochs": 3,
    "lr": 5e-5,
    "batch_size": 4,
}

NICOMPLEX_DEFAULTS = {
    "model_name": "/data/jingyuan_data/HF_models/Qwen3-4B-Instruct-2507",
    "unimol_ckpt": None,
    "unimol_dict": "/data/jingyuan_data/OMol25_MC_all/dict.txt",
    "init_ckpt": "/data/jingyuan_data/sft_intro_bridge_unimol_continue_ckpt_bos_only/checkpoint-7956",
    "output_dir": "/data/jingyuan_data/sft_nicomplex_ddg_unimol_full_ckpt",
    "lmdb": ["/data/jingyuan_data/NiComplex/data.lmdb"],
    "split_seed": 38,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "epochs": 20,
    "lr": 3e-5,
    "batch_size": 16,
}

VASKA_DEFAULTS = {
    "model_name": "/data/jingyuan_data/HF_models/Qwen3-4B-Instruct-2507",
    "unimol_ckpt": None,
    "unimol_dict": "/data/jingyuan_data/OMol25_MC_all/dict.txt",
    "init_ckpt": "/data/jingyuan_data/sft_intro_bridge_unimol_continue_ckpt_bos_only/checkpoint-7956",
    "lmdb": "/data/jingyuan_data/vaskas-space/data.lmdb",
    "output_dir": "/data/jingyuan_data/sft_vaska_barrier_unimol_full_ckpt_small_data",
    "split_seed": 38,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "epochs": 20,
    "lr": 3e-5,
    "batch_size": 32,
}
