# OOD: cluster-based train / test split (k150 far-from-train)

Same **Stage3 Property** stack (Uni-Mol [CLS] → `single_token_projection` → Qwen3, SMILES+3D prompt) as `Stage3/Property.py`.  
The only change is **which samples** are used. LMDB keys must be **CSD codes** (e.g. `ABAFOZ`).

**Split sources** (`--split_csv`):

1. **Single CSV** (default): `/home/zhujingyuan/TMC/tmQMg/cluster_split_k150_far_from_train.csv` with `CSD code` + `split` = `train` | `test`.
2. **Directory**: e.g. `split_k200_far_from_train/` with `train.csv` and `test.csv` (`CSD code` column).

## Files

| File | Role |
|------|------|
| `cluster_split.py` | Load CSV, map CSD → split |
| `dataset_ood.py` | `TmQMgClusterSplitDataset` |
| `Property_OOD.py` | Train on `train`, eval on `test` |
| `inference_property_ood.py` | Test-split MAE / R² |

## Train

```bash
deepspeed --num_gpus=2 OOD/Property_OOD.py --property dipole_moment
```

Defaults (override only if needed): `Stage2_ckpt=/data/jingyuan_data/3DTMC-LLM/Stage2`, `split_csv=cluster_split_k150_far_from_train.csv`, `epochs=10`, `save_steps=1000`.

Output: `/data/jingyuan_data/OOD_Property_<property>_<split_tag>_ckpt`  
(e.g. `..._cluster_split_k150_far_from_train_ckpt` for the default split CSV).

**All properties train + infer:** `./run_ood_property_train_infer.sh`

## Inference (cluster test split)

```bash
CUDA_VISIBLE_DEVICES=0 python -u OOD/inference_property_ood.py \
  --Stage3_ckpt /data/jingyuan_data/OOD_Property_dipole_moment_cluster_split_k150_far_from_train_ckpt/checkpoint-3000 \
  --property dipole_moment \
  --save_json /path/to/ood_test_preds.json
```

## Defaults

- **Stage2_ckpt**: `/data/jingyuan_data/3DTMC-LLM/Stage2`
- **Split (default)**: `/home/zhujingyuan/TMC/tmQMg/cluster_split_k150_far_from_train.csv`
- **epochs / save_steps**: `10` / `1000`
- **LMDBs** (both scanned, filtered by split):  
  - `/data/jingyuan_data/tmqmg/stage3/train/tmqmg_atom_only_new.lmdb`  
  - `/data/jingyuan_data/tmqmg/stage3/test/tmqmg_atom_only_new.lmdb`
