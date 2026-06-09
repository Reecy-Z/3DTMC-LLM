# Vaska OOD: 26 leave-one-ligand-out folds

Hold out one ligand from `ligand_a1` / `ligand_a2` / `ligand_b` / `ligand_c` slots; train on all complexes that do **not** contain that ligand.

Ligand list: `OOD/Vaska/ligand_split.py` → `LIGANDS` (26 names).

## Train

```bash
deepspeed --num_gpus=2 OOD/Vaska/Vaska_Ligand_OOD.py --holdout_ligand dft-co
deepspeed --num_gpus=2 OOD/Vaska/Vaska_Ligand_OOD.py --run_all_loops
```

Output: `/data/jingyuan_data/Vaska_Ligand_OOD_Models/ligand_<name>/`

## Inference

```bash
python inference.py stage3 \
  --task vaska_barrier \
  --holdout_ligand dft-co \
  --lmdb /data/jingyuan_data/vaskas-space/data.lmdb \
  --Stage3_ckpt /data/jingyuan_data/Vaska_Ligand_OOD_Models/ligand_dft-co \
  --save_json ood_preds.json
```

All 26 folds: `./run_vaska_ligand_ood_train_infer.sh`
