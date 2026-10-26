#!/usr/bin/env bash
set -euo pipefail

# Export the exact ScanNet++ K=1 / regis3 Phase-3 checkpoint into the same
# best_feature_pairs layout used by Direction_Ablation_k1 modes.
#
# Default output:
#   exp/scannetpp/Direction_Ablation_k1/split_phase3_k1_regis3/best_feature_pairs
#
# Example: write into an existing method-style path instead:
#   MODE_DIR=real_only FORCE=1 GPU=4 \
#     bash scripts/export_scannetpp_split_phase3_k1_regis3_features.sh
#
# WARNING: MODE_DIR=real_only with FORCE=1 replaces that method's current
# best_feature_pairs directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-4}"
MODE_DIR="${MODE_DIR:-split_phase3_k1_regis3}"
FORCE="${FORCE:-0}"
AUGMENTATION_MODE="${AUGMENTATION_MODE:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"

CKPT="${CKPT:-exp/scannetpp/split_phase3_k1/regis3/model/model_best.pth}"
CONFIG="${CONFIG:-configs/scannetpp/semseg-pt-v3m1-0-stage2-phase3-k1-cdc-aug.py}"
OUT_DIR="${OUT_DIR:-exp/scannetpp/Direction_Ablation_k1/${MODE_DIR}/best_feature_pairs}"

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [ ! -f "${CONFIG}" ]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi

args=(
  tools/export_phase3_feature_pairs.py
  --config "${CONFIG}"
  --checkpoint "${CKPT}"
  --output-dir "${OUT_DIR}"
  --regis regis3
  --batch-size "${BATCH_SIZE}"
)

if [ -n "${AUGMENTATION_MODE}" ]; then
  args+=(--augmentation-mode "${AUGMENTATION_MODE}")
fi
if [ "${FORCE}" = "1" ]; then
  args+=(--force)
fi

echo "============================================================"
echo "ScanNet++ K=1 regis3 feature-pair export"
echo "Checkpoint : ${CKPT}"
echo "Config     : ${CONFIG}"
echo "Output     : ${OUT_DIR}"
echo "GPU        : ${GPU}"
echo "Batch size : ${BATCH_SIZE}"
if [ -n "${AUGMENTATION_MODE}" ]; then
  echo "Mode       : ${AUGMENTATION_MODE} (override)"
else
  echo "Mode       : from config/checkpoint model (default cdc)"
fi
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u "${args[@]}"

echo "============================================================"
echo "Done. Feature pairs saved to: ${OUT_DIR}"
echo "============================================================"
