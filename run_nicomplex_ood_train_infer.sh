#!/usr/bin/env bash
# NiComplex OOD: train + infer five experiments sequentially (multi-GPU train, single-GPU infer).
#
# Experiments:
#   1) train_rest_test_Pybox   — ligand scaffold LOLO (test Pybox)
#   2) train_rest_test_Biox    — ligand scaffold LOLO (test Biox)
#   3) train_rest_test_Biim    — ligand scaffold LOLO (test Biim)
#   4) train_NiH+Csp2_test_Csp3 — reaction OOD (train NiH+C(sp3)-C(sp2), test C(sp3)-C(sp3))
#   5) train_NiH+Csp3_test_Csp2 — reaction OOD (train NiH+C(sp3)-C(sp3), test C(sp3)-C(sp2))
#
# Usage:
#   ./run_nicomplex_ood_train_infer.sh
#   TRAIN_GPUS=0,1 INFER_GPU=0 NUM_TRAIN_GPUS=2 ./run_nicomplex_ood_train_infer.sh
#   FORCE_RERUN=1 ./run_nicomplex_ood_train_infer.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-10}"

LMDB_PATH="${LMDB_PATH:-/data/jingyuan_data/NiComplex/data.lmdb}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/jingyuan_data/NiComplex_OOD_Models}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
INFER_GPU="${INFER_GPU:-0}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FORCE_RERUN="${FORCE_RERUN:-0}"

PYTHON="${PYTHON:-/home/zhujingyuan/anaconda3/envs/uni-mol/bin/python}"

EXPERIMENTS=(
  train_rest_test_Pybox
  train_rest_test_Biox
  train_rest_test_Biim
  train_NiH+Csp2_test_Csp3
  train_NiH+Csp3_test_Csp2
)

_fold_dir() {
  "${PYTHON}" -c "import sys; sys.path.insert(0, '${ROOT_DIR}'); from OOD.NiComplex.nicomplex_split import experiment_dirname; print(experiment_dirname(sys.argv[1]))" "$1"
}

_is_trained() {
  local fold_dir="$1"
  [[ -f "${fold_dir}/adapter_model.bin" && -f "${fold_dir}/experiment.txt" ]]
}

_has_predictions() {
  local json="$1"
  [[ -s "${json}" ]]
}

if [[ ! -f "${LMDB_PATH}" ]]; then
  echo "[ERROR] LMDB not found: ${LMDB_PATH}"
  exit 1
fi

echo "[INFO] LMDB=${LMDB_PATH}"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] train GPUs=${TRAIN_GPUS} (num=${NUM_TRAIN_GPUS}) | infer GPU=${INFER_GPU}"
echo "[INFO] epochs=${EPOCHS} batch_size=${BATCH_SIZE} save_steps=${SAVE_STEPS} | FORCE_RERUN=${FORCE_RERUN}"
echo "[INFO] ===== NiComplex OOD sequential train+infer start: $(date '+%F %T') ====="

for experiment in "${EXPERIMENTS[@]}"; do
  fold_name="$(_fold_dir "${experiment}")"
  fold_dir="${OUTPUT_DIR}/${fold_name}"
  pred_json="${fold_dir}/ood_test_predictions.json"

  echo
  echo "[INFO] ===== experiment=${experiment} train start: $(date '+%F %T') ====="

  if [[ "${FORCE_RERUN}" != "1" ]] && _is_trained "${fold_dir}"; then
    echo "[INFO] train skip: ${experiment} (already trained: ${fold_dir})"
  else
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" deepspeed --num_gpus="${NUM_TRAIN_GPUS}" \
      "${ROOT_DIR}/OOD/NiComplex/NiComplex_OOD.py" \
      --experiment "${experiment}" \
      --lmdb "${LMDB_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --save_steps "${SAVE_STEPS}"
    echo "[INFO] ===== experiment=${experiment} train done: $(date '+%F %T') ====="
    echo "[INFO] Cooling down ${COOLDOWN_SECONDS}s before inference..."
    sleep "${COOLDOWN_SECONDS}"
  fi

  if ! _is_trained "${fold_dir}"; then
    echo "[ERROR] No trained model for ${experiment}: ${fold_dir}"
    exit 1
  fi

  echo "[INFO] ===== experiment=${experiment} inference start: $(date '+%F %T') ====="

  if [[ "${FORCE_RERUN}" != "1" ]] && _has_predictions "${pred_json}"; then
    echo "[INFO] infer skip: ${experiment} (predictions exist: ${pred_json})"
  else
    CUDA_VISIBLE_DEVICES="${INFER_GPU}" "${PYTHON}" -u "${ROOT_DIR}/OOD/NiComplex/inference_nicomplex_ood.py" \
      --experiment "${experiment}" \
      --Stage3_ckpt "${fold_dir}" \
      --lmdb "${LMDB_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      --save_json "${pred_json}" \
      --print_every 20
    echo "[INFO] inference JSON saved: ${pred_json}"
  fi

  echo "[INFO] ===== experiment=${experiment} done: $(date '+%F %T') ====="
  echo "[INFO] Cooling down ${COOLDOWN_SECONDS}s before next experiment..."
  sleep "${COOLDOWN_SECONDS}"
done

"${PYTHON}" - <<PY
import json
import sys
from pathlib import Path

sys.path.insert(0, "${ROOT_DIR}")
from OOD.NiComplex.nicomplex_split import EXPERIMENT_NAMES, experiment_dirname

root = Path("${OUTPUT_DIR}")
rows = []
for exp in EXPERIMENT_NAMES:
    p = root / experiment_dirname(exp) / "ood_test_predictions.json"
    if not p.is_file():
        continue
    with open(p, encoding="utf-8") as f:
        obj = json.load(f)
    rows.append({
        "experiment": obj.get("experiment"),
        "split_col": obj.get("split_col"),
        "test_types": obj.get("test_types"),
        "mae": obj.get("mae"),
        "r2": obj.get("r2"),
        "n_parsed": obj.get("n_parsed"),
        "path": str(p),
    })
if rows:
    out = root / "ood_nicomplex_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    maes = [r["mae"] for r in rows if r.get("mae") is not None]
    print(f"[INFO] Summary ({len(rows)} experiments): {out}")
    if maes:
        print(f"[INFO] mean MAE across completed experiments: {sum(maes)/len(maes):.4f}")
else:
    print("[WARN] No ood_test_predictions.json found for summary.")
PY

echo "[INFO] ===== Finished NiComplex OOD train+infer: $(date '+%F %T') ====="
