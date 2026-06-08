# Ablation experiments

## K=8 learnable structure tokens (`property_k8.py`)

Compares **single CLS token** (`Stage3/Property.py`) vs **8 learnable query tokens** that cross-attend to Uni-Mol **atom** hidden states (default in [3D-MoLM](https://github.com/lsh0520/3D-MoLM), `num_query_token=8`).

## Files

| File | Role |
|------|------|
| `k8_projector.py` | `K8QueryProjector`, `build_embeds_k8` |
| `multimodal_k8.py` | `MultimodalModelK8`, `MultimodalK8Trainer` |
| `property_k8.py` | Property SFT entry (same data as `Stage3/Property.py`) |

Checkpoint writes `k8_projection.pt` (not `single_token_projection.pt`).

**Default init (fair ablation vs single-token path):**

| Module | Default in `property_k8.py` |
|--------|---------------------------|
| K8 projector | from scratch |
| LoRA (LLM) | **trainable** by default (`--train_lora`); weights **from scratch** unless `--lora_init pretrained` |
| 3D encoder | `--3D_encoder_ckpt` or `Stage2_ckpt/3D_encoder.pt` |
| Qwen base weights | pretrained HF (`--model_name`) |

Base Qwen weights are still the public checkpoint; only the **LoRA adapter** is not loaded from Stage2 unless you pass `--lora_init pretrained`.

## Run

From repo root:

```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 ablation_experiments/property_k8.py \
  --3D_encoder_ckpt /path/to/Stage2/3D_encoder.pt \
  --train_lmdb /data/jingyuan_data/tmqmg/stage3/train/tmqmg_atom_only_new.lmdb \
  --property dipole_moment \
  --save_steps 1000 \
  --output_dir /path/to/Stage3_Property_dipole_moment_k8_ckpt
```

`--Stage2_ckpt /path/to/Stage2` is shorthand (only reads `3D_encoder.pt` in that folder).

K8 projection and LoRA are from scratch by default; **not** the Stage2 LoRA adapter.

To also reuse Stage2 LoRA (same as main Property.py):

```bash
... property_k8.py --lora_init pretrained
```

## 3D-only, single-token, no SMILES (`property_3d_only.py`)

Standard stack: Uni-Mol **[CLS] → `single_token_projection` → Qwen3** (same as main method), but the **text prompt has no SMILES**—only a short instruction; structure enters via the `object_ref` 3D slot.

| File | Role |
|------|------|
| `dataset_3d_only.py` | `TmQMg3DOnlyUnimolDataset`, `INSTRUCTION_3D_ONLY` |
| `property_3d_only.py` | Train (`single_token_projection.pt` + LoRA + 3D encoder) |
| `inference_property_3d_only.py` | Test LMDB MAE / R² |

Checkpoint dir: `/data/jingyuan_data/Stage3_Property_<property>_3d_only_ckpt`

**Default trainable (same fair-init as K8):** 3D encoder + `single_token_projection` + LoRA (from scratch); Qwen base frozen (4bit). Use `--no-train_3d_encoder` / `--no-train_projection` / `--no-train_lora` to freeze.

Supports all three properties: `dipole_moment`, `polarisability`, `homo_lumo_gap`.

### Train

```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 ablation_experiments/property_3d_only.py \
  --model_name /data/jingyuan_data/HF_models/Qwen3-4B-Instruct-2507 \
  --3D_encoder_ckpt /data/jingyuan_data/checkpoint-unimol-OMol-all-500000 \
  --train_lmdb /data/jingyuan_data/tmqmg/stage3/train/tmqmg_atom_only_new.lmdb \
  --property dipole_moment \
  --save_steps 1000 --epochs 10
```

### Test

```bash
CUDA_VISIBLE_DEVICES=0 python -u ablation_experiments/inference_property_3d_only.py \
  --Stage3_ckpt /data/jingyuan_data/Stage3_Property_dipole_moment_3d_only_ckpt/checkpoint-3000 \
  --test_lmdb /data/jingyuan_data/tmqmg/stage3/test/tmqmg_atom_only_new.lmdb \
  --property dipole_moment \
  --save_json /path/to/test_preds.json
```
