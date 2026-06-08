#!/usr/bin/env bash
# OOD Property: train + cluster-test inference for all three properties.
# Split: cluster_split_k150_far_from_train.csv. Only --property changes per run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOLDOWN_SECONDS=10

SPLIT_SOURCE="/home/zhujingyuan/TMC/tmQMg/cluster_split_k150_far_from_train.csv"
SPLIT_TAG="cluster_split_k150_far_from_train"
OOD_CKPT_ROOT="/data/jingyuan_data"

PROPERTIES=(dipole_moment homo_lumo_gap polarisability)

for property in "${PROPERTIES[@]}"; do
  echo "[INFO] ===== property=${property} train start: $(date '+%F %T') ====="

  CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 "${ROOT_DIR}/OOD/Property_OOD.py" \
    --property "${property}" \
    --split_csv "${SPLIT_SOURCE}"

  OUTPUT_DIR="${OOD_CKPT_ROOT}/OOD_Property_${property}_${SPLIT_TAG}_ckpt"
  CKPT_DIR="$(ls -1d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -n 1)"
  if [[ -z "${CKPT_DIR}" || ! -d "${CKPT_DIR}" ]]; then
    echo "[ERROR] No checkpoint found under ${OUTPUT_DIR}"
    exit 1
  fi

  PRED_JSON="${CKPT_DIR}/ood_${property}_test.json"
  echo "[INFO] property=${property} inference with checkpoint: ${CKPT_DIR}"

  CUDA_VISIBLE_DEVICES=0 python -u "${ROOT_DIR}/OOD/inference_property_ood.py" \
    --Stage3_ckpt "${CKPT_DIR}" \
    --property "${property}" \
    --split_csv "${SPLIT_SOURCE}" \
    --save_json "${PRED_JSON}" \
    --print_every 50

  echo "[INFO] property=${property} inference JSON saved: ${PRED_JSON}"
  echo "[INFO] ===== property=${property} done: $(date '+%F %T') ====="
  echo "[INFO] Cooling down ${COOLDOWN_SECONDS}s before next property..."
  sleep "${COOLDOWN_SECONDS}"
done

echo "[INFO] Finished OOD train+infer for: ${PROPERTIES[*]} (split=${SPLIT_TAG})"
