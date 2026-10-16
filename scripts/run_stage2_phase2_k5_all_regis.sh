#!/usr/bin/env bash
set -euo pipefail

# Stage-2 Phase-2 launcher: independently adapt CDC for all five K=5
# registration splits. Each run starts from its matching Phase-1 checkpoint;
# no Phase-2 state is ever carried from one registration split to another.
# At the end, per-regis metrics are consolidated and mean/std statistics are
# written automatically.

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,3,5}"
RUN_ROOT="${RUN_ROOT:-stage2_phase2_k5_all_regis_cdc_adapt}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

# Preferred layout: one Phase-1 K=5 experiment containing all five checkpoints.
# Override this variable if your Phase-1 experiment has a different name.
PHASE1_EXP_ROOT="${PHASE1_EXP_ROOT:-./exp/scannet/stage2_phase1_k5_regis1_frozen_cdc}"

CONFIG="semseg-pt-v3m1-0-stage2-phase2-k5-cdc-adapt"
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
  local candidates=(
    "${PHASE1_EXP_ROOT}/model/${regis}_model_last.pth"
    "./exp/scannet/stage2_phase1_k5_all_regis_frozen_cdc/model/${regis}_model_last.pth"
    "./exp/scannet/stage2_phase1_k5_frozen_cdc/model/${regis}_model_last.pth"
    "./exp/scannet/stage2_phase1_k5_${regis}_frozen_cdc/model/${regis}_model_last.pth"
    "./exp/scannet/stage2_phase1_k5_regis1_frozen_cdc/model/${regis}_model_last.pth"
  )

  local path
  for path in "${candidates[@]}"; do
    if [ -f "${path}" ]; then
      printf '%s' "${path}"
      return 0
    fi
  done

  echo "[ERROR] Cannot find K=5 Phase-1 checkpoint for ${regis}." >&2
  echo "        Checked:" >&2
  for path in "${candidates[@]}"; do
    echo "        - ${path}" >&2
  done
  echo "        Set PHASE1_EXP_ROOT to the experiment containing all five Phase-1 checkpoints." >&2
  return 1
}

echo "============================================================"
echo "Stage-2 Phase-2 CDC adaptation: K=5, all 5 registration sets"
echo "Config      : ${CONFIG}"
echo "GPU list    : ${GPU_LIST} (${NUM_GPUS} GPUs/run)"
echo "Run root    : ./exp/${DATASET}/${RUN_ROOT}"
echo "Phase1 root : ${PHASE1_EXP_ROOT}"
echo "============================================================"

# Preflight: require all five Phase-1 checkpoints before starting any training.
for i in "${!REGIS[@]}"; do
  regis="${REGIS[$i]}"
  ckpt="$(resolve_phase1_ckpt "${regis}")"
  CKPTS+=("${ckpt}")
  echo "[FOUND] ${regis}: ${ckpt}"
done

# Refuse accidental overwrite unless explicitly allowed.
for regis in "${REGIS[@]}"; do
  out_dir="./exp/${DATASET}/${RUN_ROOT}/${regis}"
  if [ -d "${out_dir}" ] && [ "${ALLOW_EXISTING}" != "1" ]; then
    echo "[ERROR] Output already exists: ${out_dir}"
    echo "        Re-run with ALLOW_EXISTING=1 only if overwrite/resume behavior is intended."
    exit 1
  fi
done

ROOT_DIR="./exp/${DATASET}/${RUN_ROOT}"
MODEL_DIR="${ROOT_DIR}/model"
METRICS_DIR="${ROOT_DIR}/metrics"
mkdir -p "${MODEL_DIR}" "${METRICS_DIR}"

# Run the five registrations sequentially. Each run is a fresh train.sh process
# and therefore receives no model/optimizer state from the previous registration.
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
    -o "phase1_weight='${ckpt}'" \
    -o "phase2_regis_name='${regis}'" \
    -o "data.train.k_shot=5" \
    -o "data.train.seed=${seed}"

  produced="./exp/${DATASET}/${exp_name}/model/model_last.pth"
  if [ ! -f "${produced}" ]; then
    echo "[ERROR] ${regis} finished without expected checkpoint: ${produced}"
    exit 1
  fi

  consolidated="${MODEL_DIR}/${regis}_model_last.pth"
  cp -f "${produced}" "${consolidated}"

  metrics_src="./exp/${DATASET}/${exp_name}/phase2_metrics_final.json"
  history_src="./exp/${DATASET}/${exp_name}/phase2_metrics_history.json"
  if [ ! -f "${metrics_src}" ]; then
    echo "[ERROR] ${regis} finished without expected metrics: ${metrics_src}"
    exit 1
  fi
  cp -f "${metrics_src}" "${METRICS_DIR}/${regis}_final.json"
  if [ -f "${history_src}" ]; then
    cp -f "${history_src}" "${METRICS_DIR}/${regis}_history.json"
  fi

  echo "[DONE] ${regis}: checkpoint=${consolidated} metrics=${METRICS_DIR}/${regis}_final.json"
done

SUMMARY_JSON="${ROOT_DIR}/phase2_metrics_summary.json"
SUMMARY_CSV="${ROOT_DIR}/phase2_metrics_summary.csv"

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

preferred = [
    "loss",
    "tangent_loss",
    "discrimination_loss",
    "train_tas",
    "train_gdg",
    "base_tas",
    "base_gdg",
    "novel_tas",
    "novel_gdg",
    "num_base_anchors",
    "num_novel_anchors",
    "basis_orth_error",
    "cdc_factor_energy",
]
metric_keys = [
    key
    for key in preferred
    if all(isinstance(record.get(key), (int, float)) for record in records)
]

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

fieldnames = ["regis", "epoch"] + metric_keys
with open(summary_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow({key: record.get(key, "") for key in fieldnames})
    writer.writerow(
        {
            "regis": "mean",
            "epoch": "",
            **{key: aggregate[key]["mean"] for key in metric_keys},
        }
    )
    writer.writerow(
        {
            "regis": "std",
            "epoch": "",
            **{key: aggregate[key]["std"] for key in metric_keys},
        }
    )

print("\n============= Phase-2 K=5, 5-regis summary =============")
for key in (
    "loss",
    "train_tas",
    "train_gdg",
    "base_tas",
    "base_gdg",
    "novel_tas",
    "novel_gdg",
):
    if key in aggregate:
        item = aggregate[key]
        print(f"{key:22s}: {item['mean']:.6f} +/- {item['std']:.6f}")
print(f"JSON: {summary_json}")
print(f"CSV : {summary_csv}")
print("==========================================================")
PY

echo ""
echo "============================================================"
echo "All five K=5 Phase-2 CDC adaptations completed."
echo "Consolidated checkpoints:"
for regis in "${REGIS[@]}"; do
  echo "  ${MODEL_DIR}/${regis}_model_last.pth"
done
echo "Metrics:"
echo "  ${SUMMARY_JSON}"
echo "  ${SUMMARY_CSV}"
echo "============================================================"
