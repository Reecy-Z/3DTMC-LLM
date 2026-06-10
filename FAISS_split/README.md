# Train / test split with FAISS (`split_train_test_faiss.py`)

Cluster-based **train / test** splits for transition-metal complexes using **ComplexFingerprint** and **FAISS spherical K-means**.

The default strategy is **`far_from_train`**: entire clusters are held out for test. Test clusters are chosen whose centroids lie far from the dataset centroid (weighted by cluster size), so the test set tends to sit toward the edge of fingerprint space.

---

## 1. Environment

Use the **`tmc-split`** conda env (`faiss-cpu`, `numpy<2`, RDKit):

```bash
cd FAISS_split
conda env create -f environment-split.yml   # first time
conda activate tmc-split
```

To refresh an existing env:

```bash
conda env update -f environment-split.yml --prune
```

---

## 2. Input CSV

The input CSV must include at least:

| Column | Description |
|--------|-------------|
| ID column (e.g. `CSD code`) | Unique key; used to align precomputed fingerprints |
| SMILES column (e.g. `smiles`) | Dative SMILES containing a transition metal |

Example (`tmQMg/all.csv`):

```csv
CSD code,smiles,polarisability,homo_lumo_gap,dipole_moment
ABAFOZ,Cc1cc2ccccc2c2-c3c([O-]->[Pd+2]4...
```

---

## 3. (Recommended) Precomputed fingerprints

For large tables (e.g. tmQMg ~60k rows), precompute fingerprints once so each split run does not repeat RDKit work.

Pass a `.npz` file to `--fingerprints-npz`. It must contain:

- `csd_codes` — array of ID strings  
- `vectors` — fingerprint vectors aligned with `csd_codes`  
- `metal_symbol`, `formal_charge`, `d_electrons` — per-row metadata  

Example path: `/data/jingyuan_data/tmQMg_complex_fingerprints.npz`

If you omit `--fingerprints-npz`, the script builds fingerprints from SMILES on the fly (fine for small sets / debugging, slow at scale).

---

## 4. Running a split

### 4.1 Full tmQMg example (recommended)

```bash
conda activate tmc-split

python split_train_test_faiss.py \
  --input tmQMg/all.csv \
  --id-col "CSD code" \
  --smiles-col smiles \
  --fingerprints-npz /data/jingyuan_data/tmQMg_complex_fingerprints.npz \
  --n-clusters 150 \
  --kmeans-niter 25 \
  --test-frac-min 0.10 \
  --test-frac-max 0.20 \
  --w-bits 0.7 \
  --w-meta 0.3 \
  --seed 42 \
  --output tmQMg/cluster_split_k150_far_from_train.csv \
  --export-dir tmQMg/split_k150_far_from_train
```

### 4.2 Without precomputed fingerprints (small data / debug)

```bash
python split_train_test_faiss.py \
  --input your_data.csv \
  --id-col CSD_code \
  --smiles-col SMILES \
  --export-dir output/train_test
```

### 4.3 Quick debug run

```bash
python split_train_test_faiss.py \
  --input tmQMg/all.csv \
  --id-col "CSD code" \
  --smiles-col smiles \
  --fingerprints-npz /data/jingyuan_data/tmQMg_complex_fingerprints.npz \
  --max-samples 1000 \
  --export-dir /tmp/split_debug
```

---

## 5. Main CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | — | Input CSV path |
| `--id-col` | `CSD_code` | Unique ID column |
| `--smiles-col` | `SMILES_CSD_fixed` | SMILES column |
| `--fingerprints-npz` | none | Precomputed `.npz` (strongly recommended at scale) |
| `--n-clusters` | `150` | FAISS K-means cluster count |
| `--kmeans-niter` | `25` | K-means iterations |
| `--test-frac-min` | `0.10` | Minimum test fraction |
| `--test-frac-max` | `0.20` | Maximum test fraction |
| `--target-test-frac` | none | If above max, subsample test toward this fraction |
| `--w-bits` | `0.7` | Weight for 2D fingerprint block (Morgan / MACCS / RDKit) |
| `--w-meta` | `0.3` | Weight for metal / charge / d-electron meta features |
| `--seed` | `42` | Random seed (K-means and subsampling) |
| `--output` | `tmQM_faiss_cluster_split.csv` | Summary table with `cluster_id` and `split` |
| `--export-dir` | none | Directory for `train.csv` and `test.csv` |

---

## 6. Outputs

With `--export-dir tmQMg/split_k150_far_from_train`:

```
tmQMg/
├── cluster_split_k150_far_from_train.csv   # summary: ID, SMILES, metal fields, cluster_id, split
├── cluster_split_k150_far_from_train.json  # run metadata (clusters, fractions, args)
└── split_k150_far_from_train/
    ├── train.csv                           # training rows (all original input columns)
    └── test.csv                            # test rows
```

**`cluster_split_*.csv` columns:**

- ID column (e.g. `CSD code`)
- `smiles`
- `metal_symbol`, `formal_charge`, `d_electrons`
- `cluster_id` — FAISS cluster index (`0` … `n_clusters - 1`)
- `split` — `train` or `test`

**Split properties:**

- All samples in a given `cluster_id` share the same split (no cluster is mixed across train and test).
- Test clusters are selected in fingerprint space **far from the global centroid**.

Use the summary CSV or the `train.csv` / `test.csv` directory with Stage 3 property OOD training (`OOD/property/Property_OOD.py`) and `inference.py stage3 --split_csv ...`.

---

## 7. Optional: add a `Split` column to a master table

```python
import pandas as pd

all_df = pd.read_csv("tmQMg/all.csv")
train_ids = set(pd.read_csv("tmQMg/train.csv")["CSD code"].astype(str))
test_ids = set(pd.read_csv("tmQMg/test.csv")["CSD code"].astype(str))

split_map = {k: "train" for k in train_ids}
split_map.update({k: "test" for k in test_ids})
all_df.insert(1, "Split", all_df["CSD code"].astype(str).map(split_map))
all_df.to_csv("tmQMg/all.csv", index=False)
```

---

## 8. UMAP visualization (optional)

`plot_tmqmg_split_umap.py` projects the **same clustering features** used by FAISS (block-wise L2 normalization and weights) for train/test inspection:

```bash
conda activate tmc-split

python plot_tmqmg_split_umap.py \
  --split-csv tmQMg/cluster_split_k150_far_from_train.csv \
  --fit-subsample 0 \
  --contrast-only
```

`--split-json` and `--output-prefix` default to `<split-csv>.json` and `<split-csv-stem>_umap` beside the CSV.

- Do **not** pass `--raw` if you want features consistent with the split (default clustering features).
- `--contrast-only` — single figure: faint train points, highlighted test.

---

## 9. Pipeline overview

```
input.csv
   │
   ├─► (optional) precomputed .npz  ──►  --fingerprints-npz
   │
   └─► split_train_test_faiss.py
           │
           ├─ ComplexFingerprint → vector_for_clustering (L2 norm)
           ├─ FAISS spherical K-means (K = n_clusters)
           ├─ far_from_train test-cluster selection
           │
           ├─► cluster_split_*.csv / .json
           └─► train.csv + test.csv
```

---

## 10. FAQ

**`FAISS not available`**

Run inside `tmc-split` and confirm `numpy<2`:

```bash
conda activate tmc-split
python -c "import faiss; print(faiss.__version__)"
```

**Some rows are skipped**

- SMILES fails to parse or has no transition metal (on-the-fly fingerprinting).
- ID missing from the `.npz` (precomputed fingerprints).

Check the terminal `skipped` count; the summary table only includes valid rows.

**Test fraction is not exactly 10%**

By default, splits are chosen **by whole clusters** within `[test_frac_min, test_frac_max]`. The exact fraction depends on cluster sizes. Use `--target-test-frac 0.10` to nudge toward a target when within the cap.

**Changing K**

Set `--n-clusters` (e.g. 100, 150, 200) and use distinct `--output` / `--export-dir` paths to avoid overwriting previous runs.
