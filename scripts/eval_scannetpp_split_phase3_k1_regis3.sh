#!/usr/bin/env bash
set -euo pipefail

# Evaluate the exact ScanNet++ K=1 / regis3 Phase-3 checkpoint and keep the
# evaluation artifacts under Direction_Ablation_k1 without touching any of the
# existing direction-ablation runs.
#
# Usage:
#   GPU=4 bash scripts/eval_scannetpp_split_phase3_k1_regis3.sh
#
# Optional overrides:
#   GPU=0
#   CKPT=/path/to/model_best.pth
#   OUT_DIR=/path/to/output

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-4}"

CKPT="${CKPT:-./exp/scannetpp/split_phase3_k1/regis3/model/model_best.pth}"
CONFIG="${CONFIG:-./configs/scannetpp/semseg-pt-v3m1-0-stage2-phase3-k1-cdc-aug.py}"
OUT_DIR="${OUT_DIR:-./exp/scannetpp/Direction_Ablation_k1/split_phase3_k1_regis3_eval}"

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [ ! -f "${CONFIG}" ]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# Preserve exact provenance of this evaluation.
printf '%s\n' "${CKPT}" > "${OUT_DIR}/source_checkpoint.txt"
printf '%s\n' "${CONFIG}" > "${OUT_DIR}/source_config.txt"

cat > "${OUT_DIR}/evaluation_command.txt" <<EOF
CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON_BIN} -u visilization/test_gfs_vis.py \\
  --config-file ${CONFIG} \\
  --num-gpus 1 \\
  --options save_path=${OUT_DIR} weight=${CKPT} phase3_regis_name=regis3 empty_cache=True
EOF

echo "============================================================"
echo "ScanNet++ K=1 / regis3 checkpoint evaluation"
echo "Checkpoint : ${CKPT}"
echo "Config     : ${CONFIG}"
echo "Output     : ${OUT_DIR}"
echo "GPU        : ${GPU}"
echo "============================================================"

# SafeGFSSemSegTester writes evaluation_metrics.json into OUT_DIR.  No per-scene
# visualization arrays are requested here, so this run only performs validation
# and stores metrics/logs.
CUDA_VISIBLE_DEVICES="${GPU}" \
"${PYTHON_BIN}" -u visilization/test_gfs_vis.py \
  --config-file "${CONFIG}" \
  --num-gpus 1 \
  --options \
    save_path="${OUT_DIR}" \
    weight="${CKPT}" \
    phase3_regis_name="regis3" \
    empty_cache=True \
  2>&1 | tee "${OUT_DIR}/evaluation.log"

METRIC_FILE="${OUT_DIR}/evaluation_metrics.json"
if [ ! -f "${METRIC_FILE}" ]; then
  echo "[ERROR] Evaluation finished without ${METRIC_FILE}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${METRIC_FILE}" "${OUT_DIR}/summary.txt" <<'PY'
import json
import sys

metric_path, summary_path = sys.argv[1:3]
with open(metric_path, "r", encoding="utf-8") as f:
    m = json.load(f)

keys = ["mIoU", "base_mIoU", "novel_mIoU", "hm_mIoU", "mAcc", "allAcc"]
lines = ["ScanNet++ K=1 regis3 evaluation", ""]
for key in keys:
    value = m.get(key)
    lines.append(f"{key}: {value:.6f}" if isinstance(value, (int, float)) else f"{key}: {value}")
lines.append(f"num_scenes: {m.get('num_scenes')}")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
PY

echo "============================================================"
echo "Done. Results saved to: ${OUT_DIR}"
echo "Metrics : ${OUT_DIR}/evaluation_metrics.json"
echo "Summary : ${OUT_DIR}/summary.txt"
echo "Log     : ${OUT_DIR}/evaluation.log"
echo "============================================================"
