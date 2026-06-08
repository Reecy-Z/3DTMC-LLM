#!/usr/bin/env bash
# Vaska OOD: LOBO train (5 b_groups) + OOD inference on each held-out group, save JSON.
#
# Usage (from repo root):
#   ./run_vaska_ood_train_infer.sh
#
# Override GPUs:
#   TRAIN_GPUS=2,3 INFER_GPU=2 ./run_vaska_ood_train_infer.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-10}"

LMDB_PATH="${LMDB_PATH:-/data/jingyuan_data/vaskas-space/data.lmdb}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/jingyuan_data/Vaska_OOD_Models}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
INFER_GPU="${INFER_GPU:-2}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"

SUMMARY_JSON="${OUTPUT_DIR}/ood_b_group_summary.json"

# LMDB may be a single file (e.g. .../data.lmdb) or a directory containing data.mdb.
if [[ -f "${LMDB_PATH}" ]]; then
  : # single-file LMDB
elif [[ -f "${LMDB_PATH}/data.mdb" ]]; then
  : # directory-style LMDB
elif [[ -d "${LMDB_PATH}" ]]; then
  : # LMDB directory (lmdb.open accepts dir path)
else
  echo "[ERROR] LMDB not found: ${LMDB_PATH}"
  echo "        Expected a .lmdb file, or a directory with data.mdb inside."
  exit 1
fi

echo "[INFO] ===== Vaska OOD train (5 folds) start: $(date '+%F %T') ====="
echo "[INFO] LMDB=${LMDB_PATH}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] train GPUs: CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} (num_gpus=${NUM_TRAIN_GPUS})"

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" deepspeed --num_gpus="${NUM_TRAIN_GPUS}" \
  "${ROOT_DIR}/OOD/Vaska/Vaska_OOD.py" \
  --run_all_loops \
  --lmdb "${LMDB_PATH}" \
  --output_dir "${OUTPUT_DIR}"

echo "[INFO] Training finished. Cooling down ${COOLDOWN_SECONDS}s before inference..."
sleep "${COOLDOWN_SECONDS}"

echo "[INFO] ===== Vaska OOD inference (5 held-out groups) start: $(date '+%F %T') ====="
echo "[INFO] infer GPU: CUDA_VISIBLE_DEVICES=${INFER_GPU}"

CUDA_VISIBLE_DEVICES="${INFER_GPU}" python -u "${ROOT_DIR}/OOD/Vaska/inference_vaska_ood.py" \
  --run_all_loops \
  --lmdb "${LMDB_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_json "${SUMMARY_JSON}" \
  --print_every 20

echo "[INFO] Per-fold predictions:"
for bg in linear_pseudohalides halides chalcogen_donors nitro_nitrite alkynyl; do
  safe="${bg}"
  json="${OUTPUT_DIR}/bgroup_${safe}/ood_test_predictions.json"
  if [[ -f "${json}" ]]; then
    echo "  OK  ${json}"
  else
    echo "  MISSING  ${json}"
  fi
done

if [[ -f "${SUMMARY_JSON}" ]]; then
  echo "[INFO] Summary JSON: ${SUMMARY_JSON}"
else
  echo "[WARN] Summary JSON not found: ${SUMMARY_JSON}"
fi

echo "[INFO] ===== All Vaska OOD train+infer done: $(date '+%F %T') ====="
