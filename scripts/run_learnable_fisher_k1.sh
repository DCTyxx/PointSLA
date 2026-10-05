#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

GPU_LIST="${GPU_LIST:-0,3,5}"
PYTHON_BIN="${PYTHON_BIN:-/home/xxyu/miniforge3/envs/gfsvl/bin/python}"
RUN_ROOT="${RUN_ROOT:-./exp/scannet/gfsvl_learnable_fisher_k1}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKER="${NUM_WORKER:-6}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
CONFIG="configs/scannet/semseg-pt-v3m1-0-gfsregistrain_k1-metric-fisher-riemannian-learnable.py"
OUT_DIR="${RUN_ROOT}/learnable_fisher_riemannian"
LOG_FILE="${OUT_DIR}/launcher.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
    echo "[ERROR] GPU_LIST is empty" >&2
    exit 2
fi

if [[ -e "${OUT_DIR}" && "${ALLOW_EXISTING}" != "1" ]]; then
    if [[ -n "$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "[ERROR] Output directory is not empty: ${OUT_DIR}" >&2
        echo "        Set ALLOW_EXISTING=1 or use another RUN_ROOT." >&2
        exit 2
    fi
fi
mkdir -p "${OUT_DIR}"

{
    echo "============================================================"
    echo "GFS-VL Learnable Fisher-Riemannian"
    echo "GPUs       : ${GPU_LIST}"
    echo "Batch size : ${BATCH_SIZE}"
    echo "Config     : ${CONFIG}"
    echo "Output     : ${OUT_DIR}"
    echo "Start      : $(date '+%F %T')"
    echo "Git commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "============================================================"
} | tee "${LOG_FILE}"

set +e
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    "${PYTHON_BIN}" tools/train.py \
    --config-file "${CONFIG}" \
    --num-gpus "${#GPUS[@]}" \
    --options \
        "save_path=${OUT_DIR}" \
        "batch_size=${BATCH_SIZE}" \
        "num_worker=${NUM_WORKER}" \
        "num_worker_test=0" \
        "epoch=20" \
        "eval_epoch=20" \
        "regis_train_list=[regis1,regis2,regis3,regis4,regis5]" \
    >> "${LOG_FILE}" 2>&1
RC=$?
set -e

{
    echo
    echo "============================================================"
    echo "End       : $(date '+%F %T')"
    echo "Exit code : ${RC}"
    echo "============================================================"
} | tee -a "${LOG_FILE}"

exit "${RC}"
