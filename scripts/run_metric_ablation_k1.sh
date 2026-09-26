#!/usr/bin/env bash
set -Eeuo pipefail

# Run the four GFS-VL semantic-metric ablations with one command.
#
# Default:
#   - 4 experiments sequentially
#   - GPUs 3,4,5,6 are used in order
#   - ScanNet 1-shot
#   - full 20 epochs
#   - regis1..regis5
#   - official batch size / workers from the inherited configs
#
# Examples:
#   bash scripts/run_metric_ablation_k1.sh
#   GPU_LIST=2,3,4,5 bash scripts/run_metric_ablation_k1.sh
#   MODE=sequential GPU_LIST=0 bash scripts/run_metric_ablation_k1.sh
#   BATCH_SIZE=8 NUM_WORKER=8 bash scripts/run_metric_ablation_k1.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Match the official GFS-VL launcher behavior so local package imports
# (e.g. pointcept) resolve when tools/train.py is executed directly.
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

MODE="${MODE:-sequential}"
GPU_LIST="${GPU_LIST:-3,4,5,6}"
RUN_ROOT="${RUN_ROOT:-./exp/scannet/gfsvl_metric_k1_full}"
PYTHON_BIN="${PYTHON_BIN:-/home/xxyu/miniforge3/envs/gfsvl/bin/python}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKER="${NUM_WORKER:-}"

if [[ "${MODE}" != "parallel" && "${MODE}" != "sequential" ]]; then
    echo "[ERROR] MODE must be 'parallel' or 'sequential', got: ${MODE}" >&2
    exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

NAMES=(
    "cosine"
    "pca_mahalanobis"
    "local_pca"
    "fisher_riemannian"
)

CONFIGS=(
    "configs/scannet/semseg-pt-v3m1-0-gfsregistrain_k1-metric-cosine.py"
    "configs/scannet/semseg-pt-v3m1-0-gfsregistrain_k1-metric-pca-mahalanobis.py"
    "configs/scannet/semseg-pt-v3m1-0-gfsregistrain_k1-metric-local-pca.py"
    "configs/scannet/semseg-pt-v3m1-0-gfsregistrain_k1-metric-fisher-riemannian.py"
)

if [[ "${MODE}" == "parallel" && "${#GPUS[@]}" -lt "${#NAMES[@]}" ]]; then
    echo "[ERROR] Parallel mode needs at least 4 GPU ids." >&2
    echo "        Current GPU_LIST=${GPU_LIST}" >&2
    exit 2
fi

if [[ "${MODE}" == "sequential" && "${#GPUS[@]}" -lt 1 ]]; then
    echo "[ERROR] Sequential mode needs at least one GPU id." >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}"

echo "============================================================"
echo "GFS-VL semantic metric ablation"
echo "Mode       : ${MODE}"
echo "GPU list   : ${GPU_LIST}"
echo "Run root   : ${RUN_ROOT}"
echo "Python     : ${PYTHON_BIN}"
echo "Experiments: ${NAMES[*]}"
echo "============================================================"

check_output_dir() {
    local out_dir="$1"
    if [[ -e "${out_dir}" && "${ALLOW_EXISTING}" != "1" ]]; then
        if [[ -n "$(find "${out_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "[ERROR] Output directory is not empty: ${out_dir}" >&2
            echo "        Use a different RUN_ROOT or set ALLOW_EXISTING=1." >&2
            return 1
        fi
    fi
    mkdir -p "${out_dir}"
}

run_one() {
    local idx="$1"
    local gpu="$2"
    local name="${NAMES[$idx]}"
    local config="${CONFIGS[$idx]}"
    local out_dir="${RUN_ROOT}/${name}"
    local launcher_log="${out_dir}/launcher.log"

    check_output_dir "${out_dir}"

    local -a options
    options=(
        "save_path=${out_dir}"
        "epoch=20"
        "eval_epoch=20"
        "regis_train_list=[regis1,regis2,regis3,regis4,regis5]"
        "num_worker_test=0"
    )

    if [[ -n "${BATCH_SIZE}" ]]; then
        options+=("batch_size=${BATCH_SIZE}")
    fi
    if [[ -n "${NUM_WORKER}" ]]; then
        options+=("num_worker=${NUM_WORKER}")
    fi

    {
        echo "============================================================"
        echo "Experiment : ${name}"
        echo "GPU        : ${gpu}"
        echo "Config     : ${config}"
        echo "Output     : ${out_dir}"
        echo "Start      : $(date '+%F %T')"
        echo "Git commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "============================================================"
    } | tee "${launcher_log}"

    set +e
    CUDA_VISIBLE_DEVICES="${gpu}"         "${PYTHON_BIN}" tools/train.py         --config-file "${config}"         --num-gpus 1         --options "${options[@]}"         >> "${launcher_log}" 2>&1
    local rc=$?
    set -e

    {
        echo
        echo "============================================================"
        echo "Experiment : ${name}"
        echo "End        : $(date '+%F %T')"
        echo "Exit code  : ${rc}"
        echo "============================================================"
    } | tee -a "${launcher_log}"

    return "${rc}"
}

if [[ "${MODE}" == "sequential" ]]; then
    failed=0
    for i in "${!NAMES[@]}"; do
        gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
        echo "[LAUNCH] ${NAMES[$i]} on GPU ${gpu}"
        if ! run_one "${i}" "${gpu}"; then
            echo "[FAIL] ${NAMES[$i]}" >&2
            failed=1
        else
            echo "[DONE] ${NAMES[$i]}"
        fi
    done
    exit "${failed}"
fi

declare -a PIDS=()
declare -a PID_NAMES=()

for i in "${!NAMES[@]}"; do
    gpu="${GPUS[$i]}"
    echo "[LAUNCH] ${NAMES[$i]} on GPU ${gpu}"
    run_one "${i}" "${gpu}" &
    PIDS+=("$!")
    PID_NAMES+=("${NAMES[$i]}")
done

failed=0
echo
echo "[INFO] All four jobs launched. Waiting for completion..."

for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${PID_NAMES[$i]}"
    if wait "${pid}"; then
        echo "[DONE] ${name}"
    else
        rc=$?
        echo "[FAIL] ${name} (exit=${rc})" >&2
        failed=1
    fi
done

echo
echo "============================================================"
if [[ "${failed}" -eq 0 ]]; then
    echo "All four metric ablations completed successfully."
else
    echo "One or more metric ablations failed. Check:"
    echo "  ${RUN_ROOT}/<metric>/launcher.log"
fi
echo "============================================================"

exit "${failed}"
