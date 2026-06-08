#!/usr/bin/env bash
# Vaska ligand-OOD: full inference on iodide holdout test set using checkpoint-200.
#
# Usage:
#   ./run_vaska_ligand_iodide_ckpt200_infer.sh
#   INFER_GPU=0 ./run_vaska_ligand_iodide_ckpt200_infer.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOLDOUT_LIGAND="iodide"
LMDB_PATH="${LMDB_PATH:-/data/jingyuan_data/vaskas-space/data.lmdb}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/jingyuan_data/Vaska_Ligand_OOD_Models}"
FOLD_DIR="${OUTPUT_DIR}/ligand_${HOLDOUT_LIGAND}"
CKPT_DIR="${CKPT_DIR:-${FOLD_DIR}/checkpoint-200}"
PRED_JSON="${PRED_JSON:-${FOLD_DIR}/ood_test_predictions_checkpoint-200.json}"
INFER_GPU="${INFER_GPU:-0}"
PYTHON="${PYTHON:-python}"

if [[ ! -f "${LMDB_PATH}" && ! -d "${LMDB_PATH}" ]]; then
  echo "[ERROR] LMDB not found: ${LMDB_PATH}"
  exit 1
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT_DIR}"
  exit 1
fi

echo "[INFO] holdout_ligand=${HOLDOUT_LIGAND}"
echo "[INFO] LMDB=${LMDB_PATH}"
echo "[INFO] ckpt=${CKPT_DIR}"
echo "[INFO] save_json=${PRED_JSON}"
echo "[INFO] GPU=${INFER_GPU}"
echo "[INFO] ===== iodide checkpoint-200 full inference start: $(date '+%F %T') ====="

CUDA_VISIBLE_DEVICES="${INFER_GPU}" "${PYTHON}" -u "${ROOT_DIR}/OOD/Vaska/inference_vaska_ligand_ood.py" \
  --holdout_ligand "${HOLDOUT_LIGAND}" \
  --Stage3_ckpt "${CKPT_DIR}" \
  --lmdb "${LMDB_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_json "${PRED_JSON}" \
  --print_every 10

echo "[INFO] ===== iodide checkpoint-200 full inference done: $(date '+%F %T') ====="
echo "[INFO] predictions saved: ${PRED_JSON}"
