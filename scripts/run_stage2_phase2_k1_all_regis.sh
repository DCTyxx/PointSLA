#!/usr/bin/env bash
set -euo pipefail

# Stage-2 Phase-2 launcher: independently adapt CDC for all five K=1 registration
# splits.  Each run starts from its matching Phase-1 checkpoint; no Phase-2 state
# is ever carried from one regis split to another.

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,3,5}"
RUN_ROOT="${RUN_ROOT:-stage2_phase2_k1_all_regis_cdc_adapt}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

# Preferred layout: one Phase-1 experiment containing all five regis checkpoints.
# This default matches the experiment path currently used by the project.  The
# resolver also supports one-experiment-per-regis layouts below.
PHASE1_EXP_ROOT="${PHASE1_EXP_ROOT:-./exp/scannet/stage2_phase1_k1_regis1_frozen_cdc}"

CONFIG="semseg-pt-v3m1-0-stage2-phase2-k1-cdc-adapt"
DATASET="scannet"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS=${#GPU_ARRAY[@]}
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "[ERROR] GPU_LIST is empty"
  exit 1
fi

REGIS=(regis1 regis2 regis3 regis4 regis5)
SEEDS=(10 20 30 40 50)
CKPTS=()

resolve_phase1_ckpt() {
  local regis="$1"
  local idx="$2"
  local candidates=(
    "${PHASE1_EXP_ROOT}/model/${regis}_model_last.pth"
    "./exp/scannet/stage2_phase1_k1_${regis}_frozen_cdc/model/${regis}_model_last.pth"
    "./exp/scannet/stage2_phase1_k1_frozen_cdc/model/${regis}_model_last.pth"
    "./exp/scannet/stage2_phase1_k1_all_regis_frozen_cdc/model/${regis}_model_last.pth"
  )

  local path
  for path in "${candidates[@]}"; do
    if [ -f "${path}" ]; then
      printf '%s' "${path}"
      return 0
    fi
  done

  echo "[ERROR] Cannot find Phase-1 checkpoint for ${regis}." >&2
  echo "        Checked:" >&2
  for path in "${candidates[@]}"; do
    echo "        - ${path}" >&2
  done
  echo "        Set PHASE1_EXP_ROOT if your Phase-1 run is elsewhere." >&2
  return 1
}

echo "============================================================"
echo "Stage-2 Phase-2 CDC adaptation: K=1, all 5 registration sets"
echo "Config      : ${CONFIG}"
echo "GPU list    : ${GPU_LIST} (${NUM_GPUS} GPUs/run)"
echo "Run root    : ./exp/${DATASET}/${RUN_ROOT}"
echo "Phase1 root : ${PHASE1_EXP_ROOT}"
echo "============================================================"

# Preflight all five checkpoints before starting any expensive job.
for i in "${!REGIS[@]}"; do
  regis="${REGIS[$i]}"
  ckpt="$(resolve_phase1_ckpt "${regis}" "$i")"
  CKPTS+=("${ckpt}")
  echo "[FOUND] ${regis}: ${ckpt}"
done

# Refuse accidental overwrite unless explicitly requested.
for regis in "${REGIS[@]}"; do
  out_dir="./exp/${DATASET}/${RUN_ROOT}/${regis}"
  if [ -d "${out_dir}" ] && [ "${ALLOW_EXISTING}" != "1" ]; then
    echo "[ERROR] Output already exists: ${out_dir}"
    echo "        Re-run with ALLOW_EXISTING=1 only if overwrite/resume behavior is intended."
    exit 1
  fi
done

mkdir -p "./exp/${DATASET}/${RUN_ROOT}/model"

for i in "${!REGIS[@]}"; do
  regis="${REGIS[$i]}"
  seed="${SEEDS[$i]}"
  ckpt="${CKPTS[$i]}"
  exp_name="${RUN_ROOT}/${regis}"

  echo ""
  echo "============================================================"
  echo "[START] ${regis}  seed=${seed}"
  echo "Input : ${ckpt}"
  echo "Output: ./exp/${DATASET}/${exp_name}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  sh scripts/train.sh \
    -p "${PYTHON_BIN}" \
    -d "${DATASET}" \
    -c "${CONFIG}" \
    -n "${exp_name}" \
    -g "${NUM_GPUS}" \
    -o "phase1_weight='${ckpt}'" \
    -o "phase2_regis_name='${regis}'" \
    -o "data.train.seed=${seed}"

  produced="./exp/${DATASET}/${exp_name}/model/model_last.pth"
  if [ ! -f "${produced}" ]; then
    echo "[ERROR] ${regis} finished without expected checkpoint: ${produced}"
    exit 1
  fi

  consolidated="./exp/${DATASET}/${RUN_ROOT}/model/${regis}_model_last.pth"
  cp -f "${produced}" "${consolidated}"
  echo "[DONE] ${regis}: ${consolidated}"
done

echo ""
echo "============================================================"
echo "All five Phase-2 CDC adaptations completed."
echo "Consolidated checkpoints:"
for regis in "${REGIS[@]}"; do
  echo "  ./exp/${DATASET}/${RUN_ROOT}/model/${regis}_model_last.pth"
done
echo "============================================================"
