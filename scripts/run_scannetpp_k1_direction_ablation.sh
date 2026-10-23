#!/usr/bin/env bash
set -euo pipefail

# ScanNet++ K=1 Phase-3 augmentation-direction ablation.
#
# Fixed across all four settings:
#   - existing split_phase2_k1 checkpoint per registration;
#   - same Novel-head-only Phase-3 refinement;
#   - same optimizer/scheduler/epochs;
#   - same M=4 virtual samples for augmentation modes;
#   - same Base-only conservative P05 radius used by the split Phase-3 baseline;
#   - same virtual loss weight = 0.5.
#
# Only the direction source changes:
#   real_only       : no virtual feature
#   isotropic       : random unit direction in full D-dimensional feature space
#   random_subspace : direction in one fixed random rank-r orthonormal subspace
#   local_pca       : per-anchor same-class KNN residual PCA, top-r right singular vectors
#
# Recommended: use the same 3-GPU layout as the original split Phase-3 runs so
# per-GPU batch geometry for Local PCA matches the main experiment.
#
# Usage:
#   GPU_LIST=0,3,5 bash scripts/run_scannetpp_k1_direction_ablation.sh
#   MODES="real_only local_pca" GPU_LIST=0,3,5 bash scripts/run_scannetpp_k1_direction_ablation.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,3,5}"
MODES="${MODES:-real_only isotropic random_subspace local_pca}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
PHASE3_RADIUS="${PHASE3_RADIUS:-auto}"
PCA_KNN="${PCA_KNN:-32}"
PCA_CHUNK_SIZE="${PCA_CHUNK_SIZE:-512}"

DATASET="scannetpp"
K=1
CONFIG="semseg-pt-v3m1-0-stage2-phase3-k1-cdc-aug"
PHASE2_ROOT="./exp/${DATASET}/split_phase2_k1"
BASELINE_PHASE3_ROOT="./exp/${DATASET}/split_phase3_k1"
STAGE1_ROOT="./exp/${DATASET}/split_task_stage1_k1"
ABLATION_ROOT="./exp/${DATASET}/Direction_Ablation_k1"
REGIS=(regis1 regis2 regis3 regis4 regis5)
SEEDS=(10 20 30 40 50)

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS=${#GPU_ARRAY[@]}
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "[ERROR] GPU_LIST is empty"
  exit 1
fi

MODES="${MODES//,/ }"
read -r -a MODE_ARRAY <<< "${MODES}"
for mode in "${MODE_ARRAY[@]}"; do
  case "${mode}" in
    real_only|isotropic|random_subspace|local_pca) ;;
    *) echo "[ERROR] Unsupported mode: ${mode}"; exit 1 ;;
  esac
done

if [ ! -f "configs/${DATASET}/${CONFIG}.py" ]; then
  echo "[ERROR] Missing config: configs/${DATASET}/${CONFIG}.py"
  exit 1
fi
if [ ! -d "${PHASE2_ROOT}" ]; then
  echo "[ERROR] Missing Phase-2 root: ${PHASE2_ROOT}"
  exit 1
fi
mkdir -p "${ABLATION_ROOT}"

launch_train() {
  local exp_name="$1"
  shift 1

  echo ""
  echo "============================================================"
  echo "[LAUNCH] ${DATASET}/${CONFIG}"
  echo "Experiment : ./exp/${DATASET}/${exp_name}"
  echo "GPUs       : ${GPU_LIST} (${NUM_GPUS})"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  sh scripts/train.sh \
    -p "${PYTHON_BIN}" \
    -d "${DATASET}" \
    -c "${CONFIG}" \
    -n "${exp_name}" \
    -g "${NUM_GPUS}" \
    "$@"
}

check_output_dir() {
  local out_dir="$1"
  local expected="$2"
  local label="$3"

  if [ -f "${expected}" ] && [ "${REUSE_COMPLETED}" = "1" ]; then
    echo "[REUSE] ${label}: ${expected}"
    return 0
  fi
  if [ -d "${out_dir}" ] && [ "${ALLOW_EXISTING}" != "1" ]; then
    echo "[ERROR] Existing/incomplete output: ${out_dir}"
    echo "        Expected completion file: ${expected}"
    echo "        Remove it or set ALLOW_EXISTING=1 intentionally."
    return 2
  fi
  return 1
}

resolve_radius() {
  local regis="$1"

  if [ "${PHASE3_RADIUS}" != "auto" ]; then
    printf '%s' "${PHASE3_RADIUS}"
    return 0
  fi

  local radius_file="${BASELINE_PHASE3_ROOT}/${regis}/base_only_radius.txt"
  local baseline_log="${BASELINE_PHASE3_ROOT}/${regis}/train.log"
  local stage1_log="${STAGE1_ROOT}/${regis}/train.log"

  "${PYTHON_BIN}" - "${radius_file}" "${baseline_log}" "${stage1_log}" <<'PY'
import os
import re
import sys

radius_file, baseline_log, stage1_log = sys.argv[1:4]

if os.path.isfile(radius_file):
    text = open(radius_file, "r", encoding="utf-8").read().strip()
    if text:
        x = float(text)
        if x > 0:
            print(f"{x:.8f}", end="")
            raise SystemExit(0)

if os.path.isfile(baseline_log):
    text = open(baseline_log, "r", encoding="utf-8", errors="ignore").read()
    vals = re.findall(r"virtual_radius\s*[=:]\s*([0-9]*\.?[0-9]+)", text)
    if vals:
        x = float(vals[-1])
        if x > 0:
            print(f"{x:.8f}", end="")
            raise SystemExit(0)

if os.path.isfile(stage1_log):
    text = open(stage1_log, "r", encoding="utf-8", errors="ignore").read()
    vals = re.findall(
        r"RecommendedRef\(q=[0-9.]+\):\s*([0-9]*\.?[0-9]+)", text
    )
    if vals:
        x = float(vals[-1])
        if x > 0:
            print(f"{x:.8f}", end="")
            raise SystemExit(0)

raise SystemExit(
    "Cannot recover the split Phase-3/Base-P05 radius. "
    "Set PHASE3_RADIUS explicitly."
)
PY
}

summarize_mode() {
  local mode="$1"
  local root="${ABLATION_ROOT}/${mode}"

  "${PYTHON_BIN}" - "${root}" "${mode}" <<'PY'
import csv
import json
import os
import statistics
import sys

root, mode = sys.argv[1], sys.argv[2]
records = []
for i in range(1, 6):
    regis = f"regis{i}"
    path = os.path.join(root, regis, "phase3_metrics_final.json")
    with open(path, "r", encoding="utf-8") as f:
        rec = dict(json.load(f))
    rec["regis"] = regis
    records.append(rec)

keys = ["mIoU", "base_mIoU", "novel_mIoU", "hm_mIoU"]
aggregate = {}
for key in keys:
    vals = [float(r[key]) for r in records]
    aggregate[key] = {
        "mean": statistics.mean(vals),
        "std": statistics.pstdev(vals),
        "min": min(vals),
        "max": max(vals),
    }

payload = {
    "dataset": "scannetpp",
    "k_shot": 1,
    "direction_mode": mode,
    "phase2_source": "exp/scannetpp/split_phase2_k1",
    "protocol": "fixed split Phase2 -> Novel-head-only Phase3 direction ablation",
    "num_regis": 5,
    "std_definition": "population_std_ddof0",
    "per_regis": records,
    "aggregate": aggregate,
}
json_path = os.path.join(root, "phase3_metrics_summary.json")
csv_path = os.path.join(root, "phase3_metrics_summary.csv")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["regis"] + keys)
    writer.writeheader()
    for rec in records:
        writer.writerow({"regis": rec["regis"], **{k: rec[k] for k in keys}})
    writer.writerow({"regis": "mean", **{k: aggregate[k]["mean"] for k in keys}})
    writer.writerow({"regis": "std", **{k: aggregate[k]["std"] for k in keys}})

print(f"\n===== ScanNet++ K=1 direction={mode} =====")
for key in keys:
    a = aggregate[key]
    print(f"{key:22s}: {a['mean']:.6f} +/- {a['std']:.6f}")
print(f"JSON: {json_path}")
print(f"CSV : {csv_path}")
print("============================================================")
PY
}

summarize_all() {
  "${PYTHON_BIN}" - "${ABLATION_ROOT}" "${MODES}" <<'PY'
import csv
import json
import os
import sys

root, mode_text = sys.argv[1], sys.argv[2]
modes = mode_text.replace(',', ' ').split()
keys = ["mIoU", "base_mIoU", "novel_mIoU", "hm_mIoU"]
rows = []
for mode in modes:
    path = os.path.join(root, mode, "phase3_metrics_summary.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    row = {"mode": mode}
    for key in keys:
        row[f"{key}_mean"] = payload["aggregate"][key]["mean"]
        row[f"{key}_std"] = payload["aggregate"][key]["std"]
    rows.append(row)

json_path = os.path.join(root, "direction_ablation_summary.json")
csv_path = os.path.join(root, "direction_ablation_summary.csv")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "scannetpp",
        "k_shot": 1,
        "rows": rows,
    }, f, indent=2)
fields = ["mode"]
for key in keys:
    fields.extend([f"{key}_mean", f"{key}_std"])
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"[SUMMARY] {csv_path}")
PY
}

run_mode() {
  local mode="$1"
  local target_root="${ABLATION_ROOT}/${mode}"

  echo ""
  echo "####################################################################"
  echo "# ScanNet++ K=1 direction ablation: ${mode}"
  echo "# Phase-2 source: ${PHASE2_ROOT}"
  echo "# Output        : ${target_root}"
  echo "####################################################################"

  local i
  for i in "${!REGIS[@]}"; do
    local regis="${REGIS[$i]}"
    local seed="${SEEDS[$i]}"
    local p2_ckpt="${PHASE2_ROOT}/${regis}/model/model_last.pth"
    local out_dir="${target_root}/${regis}"
    local expected="${out_dir}/phase3_metrics_final.json"

    if [ ! -f "${p2_ckpt}" ]; then
      echo "[ERROR] Missing split Phase-2 checkpoint: ${p2_ckpt}"
      exit 1
    fi

    if check_output_dir "${out_dir}" "${expected}" "${mode} ${regis}"; then
      continue
    else
      status=$?
      if [ "${status}" -eq 2 ]; then exit 1; fi
    fi

    local radius
    radius="$(resolve_radius "${regis}")"
    echo "[SETTING] mode=${mode} ${regis} seed=${seed} radius=${radius} M=4"

    local exp_name="Direction_Ablation_k1/${mode}/${regis}"
    launch_train \
      "${exp_name}" \
      -o "phase2_weight='${p2_ckpt}'" \
      -o "phase3_regis_name='${regis}'" \
      -o "data.train.k_shot=1" \
      -o "data.train.seed=${seed}" \
      -o "model.augmentation_mode='${mode}'" \
      -o "model.random_subspace_seed=${seed}" \
      -o "model.local_pca_knn=${PCA_KNN}" \
      -o "model.local_pca_chunk_size=${PCA_CHUNK_SIZE}" \
      -o "model.virtual_samples=4" \
      -o "model.virtual_radius=${radius}" \
      -o "model.virtual_loss_weight=0.5"

    if [ ! -f "${expected}" ]; then
      echo "[ERROR] Missing Phase-3 metrics: ${expected}"
      exit 1
    fi
    printf '%s\n' "${radius}" > "${out_dir}/base_only_radius.txt"
    printf '%s\n' "${mode}" > "${out_dir}/augmentation_mode.txt"
  done

  summarize_mode "${mode}"
}

echo "============================================================"
echo "ScanNet++ K=1 Phase3 direction ablation"
echo "Phase2 root : ${PHASE2_ROOT}"
echo "Modes       : ${MODES}"
echo "GPUs/run    : ${GPU_LIST} (${NUM_GPUS})"
echo "Virtual M   : 4"
echo "Radius      : ${PHASE3_RADIUS} (auto = split Base-only P05)"
echo "PCA KNN     : ${PCA_KNN}"
echo "PCA chunk   : ${PCA_CHUNK_SIZE}"
echo "============================================================"

for mode in "${MODE_ARRAY[@]}"; do
  run_mode "${mode}"
done
summarize_all

echo ""
echo "[DONE] ScanNet++ K=1 direction ablation completed."
