# OOD evaluation splits

Property OOD code lives under **`OOD/property/`** (`Property_OOD.py`, `cluster_split.py`, `dataset_ood.py`, `ood_splits.py`).

Supported OOD protocols (see `property/ood_splits.py`):

| Task | Split | Training | Inference |
|------|-------|----------|-------------|
| **Property** | User CSV: `CSD code` + `split` (or `train.csv` / `test.csv` dir) | `OOD/property/Property_OOD.py` | `inference.py stage3 --split_csv ...` |
| **Vaska** | 26 leave-one-ligand-out folds | `OOD/Vaska/Vaska_Ligand_OOD.py` | `inference.py stage3 --task vaska_barrier --holdout_ligand ...` |
| **NiComplex** | Pybox / Biox / Biim scaffold holdouts | `OOD/NiComplex/NiComplex_OOD.py` | `inference.py stage3 --task nicomplex_ddg --ood_experiment ...` |

## Property (CSV split)

Provide `--split_csv` pointing to your train/test definition:

```bash
python inference.py stage3 \
  --task dipole_moment \
  --Stage3_ckpt /path/to/checkpoint \
  --split_csv /path/to/your_split.csv \
  --split_name test \
  --save_json preds.json
```

CSV format: columns `CSD code` (or `csd_code`) and `split` (`train` / `test`).  
Alternatively, a directory with `train.csv` and `test.csv` (CSD code lists only).

Train + infer all three properties:

```bash
SPLIT_SOURCE=/path/to/cluster_split.csv ./run_ood_property_train_infer.sh
```

Standard (non-OOD) property eval requires `--fixed_lmdb_eval` with `--test_lmdb` (see root `README.md`).

## Vaska (26 ligand OOD)

```bash
python inference.py stage3 \
  --task vaska_barrier \
  --holdout_ligand dft-co \
  --lmdb /data/jingyuan_data/vaskas-space/data.lmdb \
  --Stage3_ckpt /path/to/ligand_dft-co \
  --save_json ood_preds.json
```

All 26 folds: `--run_all_ood` (requires `--save_json` base name; writes per-ligand JSON).

Shell: `./run_vaska_ligand_ood_train_infer.sh`

## NiComplex (Pybox / Biox / Biim)

```bash
python inference.py stage3 \
  --task nicomplex_ddg \
  --ood_experiment train_rest_test_Pybox \
  --lmdb /data/jingyuan_data/NiComplex/data.lmdb \
  --Stage3_ckpt /path/to/exp_train_rest_test_Pybox \
  --save_json ood_preds.json
```

All three scaffold experiments: `--run_all_ood`.

Shell: `./run_nicomplex_ood_train_infer.sh`
