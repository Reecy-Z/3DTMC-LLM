#!/usr/bin/env bash
# Vaska LOLO: train + infer all 26 leave-one-ligand-out folds, save JSON.
# Skips ligands whose fold dir already has a finished checkpoint (adapter_model.bin).
#
# Usage:
#   ./run_vaska_ligand_ood_train_infer.sh
#   TRAIN_GPUS=2,3 INFER_GPU=2 ./run_vaska_ligand_ood_train_infer.sh
#   FORCE_RERUN=1 ./run_vaska_ligand_ood_train_infer.sh   # re-train / re-infer all
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-10}"

LMDB_PATH="${LMDB_PATH:-/data/jingyuan_data/vaskas-space/data.lmdb}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/jingyuan_data/Vaska_Ligand_OOD_Models}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
INFER_GPU="${INFER_GPU:-2}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
SAVE_STEPS="${SAVE_STEPS:-100}"
FORCE_RERUN="${FORCE_RERUN:-0}"

SUMMARY_JSON="${OUTPUT_DIR}/ood_ligand_summary.json"

_fold_dir() {
  python -c "import sys; sys.path.insert(0, '${ROOT_DIR}'); from OOD.Vaska.ligand_split import ligand_dirname; print(ligand_dirname(sys.argv[1]))" "$1"
}

_is_trained() {
  local fold_dir="$1"
  [[ -f "${fold_dir}/adapter_model.bin" && -f "${fold_dir}/holdout_ligand.txt" ]]
}

_has_predictions() {
  local json="$1"
  [[ -s "${json}" ]]
}

if [[ -f "${LMDB_PATH}" || -d "${LMDB_PATH}" ]]; then
  :
else
  echo "[ERROR] LMDB not found: ${LMDB_PATH}"
  exit 1
fi

mapfile -t LIGANDS < <(
  python -c "import sys; sys.path.insert(0, '${ROOT_DIR}'); from OOD.Vaska.ligand_split import LIGANDS; print('\n'.join(LIGANDS))"
)

echo "[INFO] LMDB=${LMDB_PATH} | output=${OUTPUT_DIR} | train GPUs=${TRAIN_GPUS} | FORCE_RERUN=${FORCE_RERUN}"
echo "[INFO] ===== Vaska ligand LOLO train start: $(date '+%F %T') ====="

for ligand in "${LIGANDS[@]}"; do
  fold_name="$(_fold_dir "${ligand}")"
  fold_dir="${OUTPUT_DIR}/${fold_name}"

  if [[ "${FORCE_RERUN}" != "1" ]] && _is_trained "${fold_dir}"; then
    echo "[INFO] train skip: ${ligand} (already trained: ${fold_dir})"
    continue
  fi

  echo "[INFO] ===== train ligand=${ligand} start: $(date '+%F %T') ====="
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" deepspeed --num_gpus="${NUM_TRAIN_GPUS}" \
    "${ROOT_DIR}/OOD/Vaska/Vaska_Ligand_OOD.py" \
    --holdout_ligand "${ligand}" \
    --lmdb "${LMDB_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps "${SAVE_STEPS}"
  echo "[INFO] ===== train ligand=${ligand} done: $(date '+%F %T') ====="
  echo "[INFO] Cooling down ${COOLDOWN_SECONDS}s..."
  sleep "${COOLDOWN_SECONDS}"
done

echo "[INFO] ===== Vaska ligand LOLO inference start: $(date '+%F %T') ====="

for ligand in "${LIGANDS[@]}"; do
  fold_name="$(_fold_dir "${ligand}")"
  fold_dir="${OUTPUT_DIR}/${fold_name}"
  pred_json="${fold_dir}/ood_test_predictions.json"

  if ! _is_trained "${fold_dir}"; then
    echo "[WARN] infer skip: ${ligand} (no trained model: ${fold_dir})"
    continue
  fi

  if [[ "${FORCE_RERUN}" != "1" ]] && _has_predictions "${pred_json}"; then
    echo "[INFO] infer skip: ${ligand} (predictions exist: ${pred_json})"
    continue
  fi

  echo "[INFO] ===== infer ligand=${ligand} ====="
  CUDA_VISIBLE_DEVICES="${INFER_GPU}" python -u "${ROOT_DIR}/OOD/Vaska/inference_vaska_ligand_ood.py" \
    --holdout_ligand "${ligand}" \
    --lmdb "${LMDB_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_json "${pred_json}" \
    --print_every 20
done

python - <<PY
import json
import os
from pathlib import Path

root = Path("${OUTPUT_DIR}")
rows = []
for d in sorted(root.glob("ligand_*")):
    p = d / "ood_test_predictions.json"
    if not p.is_file():
        continue
    with open(p, encoding="utf-8") as f:
        obj = json.load(f)
    rows.append({
        "holdout_ligand": obj.get("holdout_ligand"),
        "mae": obj.get("mae"),
        "r2": obj.get("r2"),
        "n_parsed": obj.get("n_parsed"),
        "path": str(p),
    })
if rows:
    out = root / "ood_ligand_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    maes = [r["mae"] for r in rows if r.get("mae") is not None]
    print(f"[INFO] Summary ({len(rows)} folds): {out}")
    if maes:
        print(f"[INFO] mean MAE across completed folds: {sum(maes)/len(maes):.4f}")
else:
    print("[WARN] No ood_test_predictions.json found for summary.")
PY

echo "[INFO] ===== Done: $(date '+%F %T') ====="
