#!/usr/bin/env bash
set -euo pipefail

# Stage-2 Phase-3 launcher: refine the Novel classifier independently for all
# five K=5 registration splits using each matching Phase-2 CDC checkpoint.
# No model/optimizer state is shared across registration splits.

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,3,5}"
RUN_ROOT="${RUN_ROOT:-stage2_phase3_k5_cdc_aug_v1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
PHASE2_EXP_ROOT="${PHASE2_EXP_ROOT:-./exp/scannet/stage2_phase2_k5_all_regis_cdc_adapt}"

CONFIG="semseg-pt-v3m1-0-stage2-phase3-k5-cdc-aug"
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

resolve_phase2_ckpt() {
  local regis="$1"
  local candidates=(
    "${PHASE2_EXP_ROOT}/model/${regis}_model_last.pth"
    "${PHASE2_EXP_ROOT}/${regis}/model/model_last.pth"
    "./exp/scannet/stage2_phase2_k5_all_regis_cdc_adapt/model/${regis}_model_last.pth"
    "./exp/scannet/stage2_phase2_k5_${regis}_cdc_adapt/model/model_last.pth"
  )

  local path
  for path in "${candidates[@]}"; do
    if [ -f "${path}" ]; then
      printf '%s' "${path}"
      return 0
    fi
  done

  echo "[ERROR] Cannot find K=5 Phase-2 checkpoint for ${regis}." >&2
  echo "        Checked:" >&2
  for path in "${candidates[@]}"; do
    echo "        - ${path}" >&2
  done
  echo "        Set PHASE2_EXP_ROOT if your Phase-2 run is elsewhere." >&2
  return 1
}

echo "============================================================"
echo "Stage-2 Phase-3 CDC augmentation: K=5, all 5 registration sets"
echo "Config      : ${CONFIG}"
echo "GPU list    : ${GPU_LIST} (${NUM_GPUS} GPUs/run)"
echo "Run root    : ./exp/${DATASET}/${RUN_ROOT}"
echo "Phase2 root : ${PHASE2_EXP_ROOT}"
echo "============================================================"

# Preflight all five source checkpoints before launching any expensive job.
for i in "${!REGIS[@]}"; do
  regis="${REGIS[$i]}"
  ckpt="$(resolve_phase2_ckpt "${regis}")"
  CKPTS+=("${ckpt}")
  echo "[FOUND] ${regis}: ${ckpt}"
done

# Refuse accidental overwrite unless explicitly allowed.
for regis in "${REGIS[@]}"; do
  out_dir="./exp/${DATASET}/${RUN_ROOT}/${regis}"
  if [ -d "${out_dir}" ] && [ "${ALLOW_EXISTING}" != "1" ]; then
    echo "[ERROR] Output already exists: ${out_dir}"
    echo "        Remove it or re-run with ALLOW_EXISTING=1 if intended."
    exit 1
  fi
done

ROOT_DIR="./exp/${DATASET}/${RUN_ROOT}"
MODEL_DIR="${ROOT_DIR}/model"
METRICS_DIR="${ROOT_DIR}/metrics"
mkdir -p "${MODEL_DIR}" "${METRICS_DIR}"

# Run regis1..regis5 sequentially. Each run is a fresh train.sh process and
# therefore starts strictly from its own matching Phase-2 checkpoint.
for i in "${!REGIS[@]}"; do
  regis="${REGIS[$i]}"
  seed="${SEEDS[$i]}"
  ckpt="${CKPTS[$i]}"
  exp_name="${RUN_ROOT}/${regis}"

  echo ""
  echo "============================================================"
  echo "[START] ${regis}  K=5  seed=${seed}"
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
    -o "phase2_weight='${ckpt}'" \
    -o "phase3_regis_name='${regis}'" \
    -o "data.train.k_shot=5" \
    -o "data.train.seed=${seed}"

  best_src="./exp/${DATASET}/${exp_name}/model/model_best.pth"
  last_src="./exp/${DATASET}/${exp_name}/model/model_last.pth"
  metrics_src="./exp/${DATASET}/${exp_name}/phase3_metrics_final.json"
  history_src="./exp/${DATASET}/${exp_name}/phase3_val_history.json"

  if [ ! -f "${best_src}" ]; then
    echo "[ERROR] ${regis} finished without expected best-HM checkpoint: ${best_src}"
    exit 1
  fi
  if [ ! -f "${metrics_src}" ]; then
    echo "[ERROR] ${regis} finished without expected precise metrics: ${metrics_src}"
    exit 1
  fi

  cp -f "${best_src}" "${MODEL_DIR}/${regis}_model_best.pth"
  if [ -f "${last_src}" ]; then
    cp -f "${last_src}" "${MODEL_DIR}/${regis}_model_last.pth"
  fi
  cp -f "${metrics_src}" "${METRICS_DIR}/${regis}_final.json"
  if [ -f "${history_src}" ]; then
    cp -f "${history_src}" "${METRICS_DIR}/${regis}_val_history.json"
  fi

  echo "[DONE] ${regis}: best=${MODEL_DIR}/${regis}_model_best.pth metrics=${METRICS_DIR}/${regis}_final.json"
done

SUMMARY_JSON="${ROOT_DIR}/phase3_metrics_summary.json"
SUMMARY_CSV="${ROOT_DIR}/phase3_metrics_summary.csv"

"${PYTHON_BIN}" - "${METRICS_DIR}" "${SUMMARY_JSON}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import os
import statistics
import sys

metrics_dir, summary_json, summary_csv = sys.argv[1:4]
regis_names = [f"regis{i}" for i in range(1, 6)]
records = []
for regis in regis_names:
    path = os.path.join(metrics_dir, f"{regis}_final.json")
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    if record.get("regis") != regis:
        raise RuntimeError(
            f"Metric/checkpoint split mismatch: expected {regis}, got {record.get('regis')}"
        )
    records.append(record)

metric_keys = ["mIoU", "base_mIoU", "novel_mIoU", "hm_mIoU"]
aggregate = {}
for key in metric_keys:
    values = [float(record[key]) for record in records]
    aggregate[key] = {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }

payload = {
    "k_shot": 5,
    "num_regis": len(records),
    "std_definition": "population_std_ddof0",
    "per_regis": records,
    "aggregate": aggregate,
}
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)

fieldnames = ["regis", "selected_checkpoint"] + metric_keys
with open(summary_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow({key: record.get(key, "") for key in fieldnames})
    writer.writerow(
        {
            "regis": "mean",
            "selected_checkpoint": "",
            **{key: aggregate[key]["mean"] for key in metric_keys},
        }
    )
    writer.writerow(
        {
            "regis": "std",
            "selected_checkpoint": "",
            **{key: aggregate[key]["std"] for key in metric_keys},
        }
    )

print("\n============= Phase-3 K=5, 5-regis summary =============")
for key in metric_keys:
    item = aggregate[key]
    print(f"{key:22s}: {item['mean']:.6f} +/- {item['std']:.6f}")
print(f"JSON: {summary_json}")
print(f"CSV : {summary_csv}")
print("==========================================================")
PY

echo ""
echo "============================================================"
echo "All five K=5 Phase-3 CDC augmentation runs completed."
echo "Best-HM checkpoints:"
for regis in "${REGIS[@]}"; do
  echo "  ${MODEL_DIR}/${regis}_model_best.pth"
done
echo "Metrics:"
echo "  ${SUMMARY_JSON}"
echo "  ${SUMMARY_CSV}"
echo "============================================================"
