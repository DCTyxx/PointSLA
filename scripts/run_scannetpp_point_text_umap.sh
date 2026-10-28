#!/usr/bin/env bash
set -euo pipefail

# Compare ScanNet++ PTv3 64-D point features against the learned SSC semantic
# space and frozen text prototypes.
#
# Default:
#   GPU=4 bash scripts/run_scannetpp_point_text_umap.sh
#
# Useful overrides:
#   METHOD=both GPU=4 bash scripts/run_scannetpp_point_text_umap.sh
#   MAX_POINTS_PER_CLASS=200 GPU=4 bash scripts/run_scannetpp_point_text_umap.sh
#   CHECKPOINT=/path/to/model_best.pth GPU=4 bash scripts/run_scannetpp_point_text_umap.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-4}"
METHOD="${METHOD:-auto}"
MAX_POINTS_PER_CLASS="${MAX_POINTS_PER_CLASS:-128}"
MIN_POINTS_PER_CLASS="${MIN_POINTS_PER_CLASS:-32}"
MAX_PER_SCENE_PER_CLASS="${MAX_PER_SCENE_PER_CLASS:-32}"
SEED="${SEED:-3407}"

CONFIG="${CONFIG:-configs/scannetpp/semseg-pt-v3m1-0-conservative-boundary-semantic-tangent-cdcbase-frozen-r24.py}"
CHECKPOINT="${CHECKPOINT:-exp/scannetpp/cdc_stage1_r24/model/model_best.pth}"
OUT_DIR="${OUT_DIR:-exp/scannetpp/feature_space_separability}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

cat <<EOF
============================================================
PointSLA Point/Text Feature-Space Visualization
Config      : ${CONFIG}
Checkpoint  : ${CHECKPOINT}
Output      : ${OUT_DIR}
GPU         : ${GPU}
Method      : ${METHOD}
Points/class: ${MAX_POINTS_PER_CLASS}
============================================================
EOF

CUDA_VISIBLE_DEVICES="${GPU}" \
"${PYTHON_BIN}" -u tools/run_feature_space_compat.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT_DIR}" \
  --method "${METHOD}" \
  --max-points-per-class "${MAX_POINTS_PER_CLASS}" \
  --min-points-per-class "${MIN_POINTS_PER_CLASS}" \
  --max-per-scene-per-class "${MAX_PER_SCENE_PER_CLASS}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUT_DIR}/run.log"

echo "============================================================"
echo "Done: ${OUT_DIR}"
echo "Metrics: ${OUT_DIR}/metrics.json"
echo "============================================================"
