#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-10}"
MAX_RETRIES=1
RESULTS_DIR="${ROOT_DIR}/Vaska_Complex_Results"
LMDB_PATH="${LMDB_PATH:-/data/jingyuan_data/vaskas-space/data.lmdb}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
INFER_GPU="${INFER_GPU:-0}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
FORCE_RERUN="${FORCE_RERUN:-0}"

STAGE3_OUTPUT_BASE="$(cd "${ROOT_DIR}" && python - <<'PY'
from train_defaults import VASKA_DEFAULTS
print(VASKA_DEFAULTS["output_dir"])
PY
)"

mkdir -p "${RESULTS_DIR}"

if [[ -f "${LMDB_PATH}" || -d "${LMDB_PATH}" ]]; then
  :
else
  echo "[ERROR] LMDB not found: ${LMDB_PATH}"
  exit 1
fi

echo "[INFO] LMDB=${LMDB_PATH} | train GPUs=${TRAIN_GPUS} | infer GPU=${INFER_GPU} | FORCE_RERUN=${FORCE_RERUN}"

for seed in $(seq 1 10); do
  PRED_JSON="${RESULTS_DIR}/pred_vaska_barrier_seed_${seed}.json"
  if [[ "${FORCE_RERUN}" != "1" && -s "${PRED_JSON}" ]]; then
    echo "[INFO] seed=${seed} skip: prediction JSON already exists (non-empty): ${PRED_JSON}"
    continue
  fi

  echo "[INFO] ===== seed=${seed} start: $(date '+%F %T') ====="

  attempt=0
  run_ok=0
  seed_start_ts=$(date +%s)

  while [[ "${attempt}" -le "${MAX_RETRIES}" ]]; do
    attempt=$((attempt + 1))
    echo "[INFO] seed=${seed} attempt=${attempt}"

    if CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" deepspeed --num_gpus="${NUM_TRAIN_GPUS}" \
      "${ROOT_DIR}/Stage3/Vaska_Complex.py" \
      --split_seed "${seed}" \
      --lmdb "${LMDB_PATH}"; then
      run_ok=1
      break
    fi

    if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
      echo "[WARN] seed=${seed} failed, retrying after ${COOLDOWN_SECONDS}s cooldown..."
      sleep "${COOLDOWN_SECONDS}"
    fi
  done

  seed_end_ts=$(date +%s)
  elapsed=$((seed_end_ts - seed_start_ts))

  if [[ "${run_ok}" -ne 1 ]]; then
    echo "[ERROR] seed=${seed} failed after ${attempt} attempts."
    exit 1
  fi

  CKPT_DIR="${STAGE3_OUTPUT_BASE}/seed_${seed}/checkpoint-500"
  echo "[INFO] seed=${seed} inference with checkpoint: ${CKPT_DIR}"
  CUDA_VISIBLE_DEVICES="${INFER_GPU}" python -u "${ROOT_DIR}/inference_Vaska_Complex.py" \
    --Stage3_ckpt "${CKPT_DIR}" \
    --split_seed "${seed}" \
    --lmdb "${LMDB_PATH}" \
    --save_json "${PRED_JSON}"
  echo "[INFO] seed=${seed} inference JSON saved: ${PRED_JSON}"

  echo "[INFO] ===== seed=${seed} done: $(date '+%F %T'), elapsed=${elapsed}s ====="
  echo "[INFO] Cooling down ${COOLDOWN_SECONDS}s before next seed..."
  sleep "${COOLDOWN_SECONDS}"
done

echo "[INFO] Finished all seeds 1..10"
echo "[INFO] Generating summary plots under ${RESULTS_DIR}"
python "${ROOT_DIR}/plot_vaska_barrier.py" --dir "${RESULTS_DIR}" --pattern "pred_vaska_barrier_seed_*.json"
