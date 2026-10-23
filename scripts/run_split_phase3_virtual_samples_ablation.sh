#!/usr/bin/env bash
set -euo pipefail

# Phase-3-only ablation for the number of CDC-generated virtual Novel features
# using the SPLIT protocol requested for ScanNet200.
#
# Fixed CDC sources (already completed):
#   exp/scannet200/split_phase2_k1/regis*/model/model_last.pth
#   exp/scannet200/split_phase2_k5/regis*/model/model_last.pth
#
# Existing M=4 baselines:
#   exp/scannet200/split_phase3_k1
#   exp/scannet200/split_phase3_k5
#
# Sweep ONLY:
#   model.virtual_samples = M, M in {2,4,8,16,32}
#
# Everything else remains identical to the original Phase-3 config/checkpoint:
#   - same split_phase2 CDC checkpoint per registration
#   - same virtual radius as the existing split_phase3 baseline (parsed from log)
#   - virtual_loss_weight = 0.5
#   - same optimizer / scheduler / epochs / seeds
#
# By default M=4 reuses the already completed split_phase3 result instead of
# rerunning it. Set REUSE_BASELINE_M4=0 if an explicit M=4 rerun is desired.
#
# Outputs:
#   exp/scannet200/Split_Virtual_Samples_Ablation/k<K>/M<M>/regis*/
#   exp/scannet200/Split_Virtual_Samples_Ablation/k<K>/virtual_samples_ablation_summary.{json,csv}
#
# Usage:
#   GPU_LIST=0,3,5 bash scripts/run_split_phase3_virtual_samples_ablation.sh
#   K_LIST=1 GPU_LIST=0,3,5 bash scripts/run_split_phase3_virtual_samples_ablation.sh
#   M_VALUES="2 8 16 32" bash scripts/run_split_phase3_virtual_samples_ablation.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,3,5}"
K_LIST="${K_LIST:-1 5}"
M_VALUES="${M_VALUES:-2 4 8 16 32}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
REUSE_BASELINE_M4="${REUSE_BASELINE_M4:-1}"
PHASE3_RADIUS="${PHASE3_RADIUS:-auto}"

DATASET="scannet200"
ABLATION_ROOT="./exp/${DATASET}/Split_Virtual_Samples_Ablation"
REGIS=(regis1 regis2 regis3 regis4 regis5)
SEEDS=(10 20 30 40 50)

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS=${#GPU_ARRAY[@]}
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "[ERROR] GPU_LIST is empty"
  exit 1
fi

K_LIST="${K_LIST//,/ }"
M_VALUES="${M_VALUES//,/ }"
read -r -a K_ARRAY <<< "${K_LIST}"
read -r -a M_ARRAY <<< "${M_VALUES}"

for k in "${K_ARRAY[@]}"; do
  if [ "${k}" != "1" ] && [ "${k}" != "5" ]; then
    echo "[ERROR] K_LIST supports only 1 and/or 5; got ${k}"
    exit 1
  fi
  config="configs/${DATASET}/semseg-pt-v3m1-0-stage2-phase3-k${k}-cdc-aug.py"
  if [ ! -f "${config}" ]; then
    echo "[ERROR] Missing Phase-3 config: ${config}"
    exit 1
  fi
  p2_root="./exp/${DATASET}/split_phase2_k${k}"
  p3_root="./exp/${DATASET}/split_phase3_k${k}"
  if [ ! -d "${p2_root}" ]; then
    echo "[ERROR] Missing split Phase-2 root: ${p2_root}"
    exit 1
  fi
  if [ "${REUSE_BASELINE_M4}" = "1" ] && [ ! -d "${p3_root}" ]; then
    echo "[ERROR] Missing split M=4 baseline root: ${p3_root}"
    exit 1
  fi
done

for m in "${M_ARRAY[@]}"; do
  if ! [[ "${m}" =~ ^[0-9]+$ ]] || [ "${m}" -lt 1 ]; then
    echo "[ERROR] M must be a positive integer; got ${m}"
    exit 1
  fi
done

mkdir -p "${ABLATION_ROOT}"

launch_train() {
  local config="$1"
  local exp_name="$2"
  shift 2

  echo ""
  echo "============================================================"
  echo "[LAUNCH] ${DATASET}/${config}"
  echo "Experiment : ./exp/${DATASET}/${exp_name}"
  echo "GPUs       : ${GPU_LIST} (${NUM_GPUS})"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  sh scripts/train.sh \
    -p "${PYTHON_BIN}" \
    -d "${DATASET}" \
    -c "${config}" \
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
    echo "[ERROR] Existing/incomplete output directory: ${out_dir}"
    echo "        Expected completion file: ${expected}"
    echo "        Remove it or set ALLOW_EXISTING=1 intentionally."
    return 2
  fi
  return 1
}

# Recover the exact virtual radius used by the existing split_phase3 baseline.
# This keeps the study strictly M-only. If it cannot be parsed, the script fails
# instead of silently changing rho, because the user explicitly wants the same
# split_phase2 CDC baseline with only the number of virtual samples changed.
resolve_radius() {
  local k="$1"
  local regis="$2"

  if [ "${PHASE3_RADIUS}" != "auto" ]; then
    printf '%s' "${PHASE3_RADIUS}"
    return 0
  fi

  local baseline_dir="./exp/${DATASET}/split_phase3_k${k}/${regis}"
  local radius_file="${baseline_dir}/base_only_radius.txt"
  local baseline_log="${baseline_dir}/train.log"

  "${PYTHON_BIN}" - "${radius_file}" "${baseline_log}" <<'PY'
import os
import re
import sys

radius_file, log_file = sys.argv[1], sys.argv[2]

if os.path.isfile(radius_file):
    text = open(radius_file, "r", encoding="utf-8").read().strip()
    if text:
        x = float(text)
        if x > 0:
            print(f"{x:.8f}", end="")
            raise SystemExit(0)

patterns = [
    re.compile(r"virtual_radius\s*[=:]\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"['\"]virtual_radius['\"]\s*:\s*([0-9]*\.?[0-9]+)"),
]
if os.path.isfile(log_file):
    text = open(log_file, "r", encoding="utf-8", errors="ignore").read()
    vals = []
    for pat in patterns:
        vals.extend(pat.findall(text))
    if vals:
        x = float(vals[-1])
        if x > 0:
            print(f"{x:.8f}", end="")
            raise SystemExit(0)

raise SystemExit(
    "Cannot recover split baseline virtual_radius from "
    f"{radius_file} or {log_file}. Set PHASE3_RADIUS explicitly."
)
PY
}

copy_baseline_m4_metrics() {
  local k="$1"
  local regis="$2"
  local target_dir="$3"

  local source_dir="./exp/${DATASET}/split_phase3_k${k}/${regis}"
  local source_metric="${source_dir}/phase3_metrics_final.json"
  if [ ! -f "${source_metric}" ]; then
    echo "[ERROR] M=4 split baseline metric missing: ${source_metric}"
    return 1
  fi

  mkdir -p "${target_dir}"
  cp -f "${source_metric}" "${target_dir}/phase3_metrics_final.json"
  if [ -f "${source_dir}/phase3_val_history.json" ]; then
    cp -f "${source_dir}/phase3_val_history.json" "${target_dir}/phase3_val_history.json"
  fi
  if [ -f "${source_dir}/base_only_radius.txt" ]; then
    cp -f "${source_dir}/base_only_radius.txt" "${target_dir}/base_only_radius.txt"
  fi
  printf '%s\n' "${source_dir}" > "${target_dir}/baseline_source.txt"
}

summarize_m() {
  local k="$1"
  local m="$2"
  local root="${ABLATION_ROOT}/k${k}/M${m}"

  "${PYTHON_BIN}" - "${root}" "${k}" "${m}" <<'PY'
import csv
import json
import os
import statistics
import sys

root, k, m = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
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
    "dataset": "scannet200",
    "protocol": "split_phase2 fixed CDC -> Phase3 virtual-sample ablation",
    "k_shot": k,
    "virtual_samples": m,
    "phase2_source": f"exp/scannet200/split_phase2_k{k}",
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
        writer.writerow({"regis": rec["regis"], **{key: rec[key] for key in keys}})
    writer.writerow({"regis": "mean", **{key: aggregate[key]["mean"] for key in keys}})
    writer.writerow({"regis": "std", **{key: aggregate[key]["std"] for key in keys}})

print(f"\n===== ScanNet200 split K={k}, virtual_samples M={m} =====")
for key in keys:
    a = aggregate[key]
    print(f"{key:22s}: {a['mean']:.6f} +/- {a['std']:.6f}")
print(f"JSON: {json_path}")
print(f"CSV : {csv_path}")
print("============================================================")
PY
}

summarize_sweep() {
  local k="$1"
  local root="${ABLATION_ROOT}/k${k}"

  "${PYTHON_BIN}" - "${root}" "${k}" "${M_VALUES}" <<'PY'
import csv
import json
import os
import sys

root, k, m_values = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ms = [int(x) for x in m_values.replace(',', ' ').split()]
rows = []
for m in ms:
    path = os.path.join(root, f"M{m}", "phase3_metrics_summary.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    row = {"M": m}
    for key in ["mIoU", "base_mIoU", "novel_mIoU", "hm_mIoU"]:
        row[f"{key}_mean"] = payload["aggregate"][key]["mean"]
        row[f"{key}_std"] = payload["aggregate"][key]["std"]
    rows.append(row)

json_path = os.path.join(root, "virtual_samples_ablation_summary.json")
csv_path = os.path.join(root, "virtual_samples_ablation_summary.csv")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "scannet200",
        "protocol": "split_phase2 fixed CDC -> Phase3 M-only ablation",
        "k_shot": k,
        "rows": rows,
    }, f, indent=2)
fields = [
    "M",
    "mIoU_mean", "mIoU_std",
    "base_mIoU_mean", "base_mIoU_std",
    "novel_mIoU_mean", "novel_mIoU_std",
    "hm_mIoU_mean", "hm_mIoU_std",
]
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"[SUMMARY] ScanNet200 split K={k}: {csv_path}")
PY
}

run_k_m() {
  local k="$1"
  local m="$2"
  local config="semseg-pt-v3m1-0-stage2-phase3-k${k}-cdc-aug"
  local p2_root="./exp/${DATASET}/split_phase2_k${k}"
  local target_root="${ABLATION_ROOT}/k${k}/M${m}"

  echo ""
  echo "####################################################################"
  echo "# ScanNet200 split K=${k}: virtual_samples M=${m}"
  echo "# Phase-2 source: ${p2_root}"
  echo "# Output        : ${target_root}"
  echo "####################################################################"

  local i
  for i in "${!REGIS[@]}"; do
    local regis="${REGIS[$i]}"
    local seed="${SEEDS[$i]}"
    local p2_ckpt="${p2_root}/${regis}/model/model_last.pth"
    local out_dir="${target_root}/${regis}"
    local expected="${out_dir}/phase3_metrics_final.json"

    if [ ! -f "${p2_ckpt}" ]; then
      echo "[ERROR] Missing split Phase-2 checkpoint: ${p2_ckpt}"
      exit 1
    fi

    if [ "${m}" = "4" ] && [ "${REUSE_BASELINE_M4}" = "1" ]; then
      if [ -f "${expected}" ] && [ "${REUSE_COMPLETED}" = "1" ]; then
        echo "[REUSE] M=4 split baseline ${regis}"
      else
        copy_baseline_m4_metrics "${k}" "${regis}" "${out_dir}"
        echo "[BASELINE] reused split_phase3_k${k}/${regis} for M=4"
      fi
      continue
    fi

    if check_output_dir "${out_dir}" "${expected}" "K=${k} M=${m} ${regis}"; then
      continue
    else
      status=$?
      if [ "${status}" -eq 2 ]; then exit 1; fi
    fi

    radius="$(resolve_radius "${k}" "${regis}")"
    echo "[RADIUS] K=${k} ${regis}: ${radius}"

    # scripts/train.sh prefixes ./exp/scannet200/, so this experiment name keeps
    # all outputs under the dedicated split ablation root.
    exp_name="Split_Virtual_Samples_Ablation/k${k}/M${m}/${regis}"
    launch_train \
      "${config}" \
      "${exp_name}" \
      -o "phase2_weight='${p2_ckpt}'" \
      -o "phase3_regis_name='${regis}'" \
      -o "data.train.k_shot=${k}" \
      -o "data.train.seed=${seed}" \
      -o "model.virtual_samples=${m}" \
      -o "model.virtual_radius=${radius}"

    if [ ! -f "${expected}" ]; then
      echo "[ERROR] Missing Phase-3 metrics: ${expected}"
      exit 1
    fi
    printf '%s\n' "${radius}" > "${out_dir}/base_only_radius.txt"
  done

  summarize_m "${k}" "${m}"
}

echo "============================================================"
echo "ScanNet200 split Phase3 virtual-sample ablation"
echo "Phase2 roots : split_phase2_k1 / split_phase2_k5"
echo "M values     : ${M_VALUES}"
echo "K values     : ${K_LIST}"
echo "GPUs/run     : ${GPU_LIST} (${NUM_GPUS})"
echo "M=4 baseline : reuse=${REUSE_BASELINE_M4}"
echo "============================================================"

for k in "${K_ARRAY[@]}"; do
  for m in "${M_ARRAY[@]}"; do
    run_k_m "${k}" "${m}"
  done
  summarize_sweep "${k}"
done

echo ""
echo "[DONE] Split Phase3 virtual-sample ablation completed."
