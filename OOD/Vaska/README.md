# Vaska OOD: leave-one-b_group-out

Same **Stage3 Vaska** stack as `Stage3/Vaska_Complex.py` (single-token + SMILES + 3D).  
Split is **by `b_group`** in `/data/jingyuan_data/vaskas-space/data.lmdb` (cluster OOD), not random 80/10/10.

Each fold holds out **one** of 5 `b_group` labels for OOD test; trains on the other four.

| b_group | ~count |
|---------|--------|
| `linear_pseudohalides` | 651 |
| `halides` | 551 |
| `chalcogen_donors` | 347 |
| `nitro_nitrite` | 200 |
| `alkynyl` | 198 |

## Files

| File | Role |
|------|------|
| `b_group_split.py` | LOBO split helpers |
| `Vaska_OOD.py` | Train one fold or all 5 |
| `inference_vaska_ood.py` | Eval on held-out b_group |

## Training (matches random-split Vaska defaults)

- `Stage2_ckpt`: `/data/jingyuan_data/3DTMC-LLM/Stage2`
- `epochs=20`, `batch_size=32`, `lr=3e-5`, `save_steps=50`
- `projection_init=from_scratch`, LoRA `r=32`, `alpha=64`, `target=all`
- **All non-holdout samples → train** (no val split; eval via `inference_vaska_ood.py`)

### One fold

```bash
deepspeed --num_gpus=2 OOD/Vaska/Vaska_OOD.py --holdout_b_group halides
```

### All 5 folds (sequential loop)

```bash
deepspeed --num_gpus=2 OOD/Vaska/Vaska_OOD.py --run_all_loops
```

Output per fold: `/data/jingyuan_data/Vaska_OOD_Models/bgroup_<name>/`

## Inference (OOD test = held-out b_group)

```bash
CUDA_VISIBLE_DEVICES=0 python -u OOD/Vaska/inference_vaska_ood.py \
  --holdout_b_group halides \
  --Stage3_ckpt /data/jingyuan_data/Vaska_OOD_Models/bgroup_halides/checkpoint-xxx
```

Evaluate all folds (auto-load ckpt from `Vaska_OOD_Models/bgroup_*/`):

```bash
CUDA_VISIBLE_DEVICES=0 python -u OOD/Vaska/inference_vaska_ood.py --run_all_loops
```

---

# Vaska OOD: leave-one-ligand-out (26 folds)

Split by **ligand identity** in `ligand_a1` / `ligand_a2` / `ligand_b` / `ligand_c` (26 unique ligands).

- **OOD test**: complexes where the held-out ligand appears in **any** of the four slots
- **Train**: all complexes that do **not** contain the held-out ligand

| File | Role |
|------|------|
| `ligand_split.py` | LOLO split helpers + `LIGANDS` tuple |
| `Vaska_Ligand_OOD.py` | Train one fold or all 26 |
| `inference_vaska_ligand_ood.py` | Eval on held-out ligand test set |

Example OOD test sizes (1947 samples): `dft-co` 721, `dft-hicn` 629, `iodide` 30, …

### One fold

```bash
CUDA_VISIBLE_DEVICES=2,3 deepspeed --num_gpus=2 OOD/Vaska/Vaska_Ligand_OOD.py \
  --holdout_ligand dft-co \
  --lmdb /data/jingyuan_data/vaskas-space/data.lmdb
```

### All 26 folds + inference

```bash
./run_vaska_ligand_ood_train_infer.sh
```

Output per fold: `/data/jingyuan_data/Vaska_Ligand_OOD_Models/ligand_<name>/ood_test_predictions.json`  
Summary: `/data/jingyuan_data/Vaska_Ligand_OOD_Models/ood_ligand_summary.json`
