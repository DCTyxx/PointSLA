#!/usr/bin/env bash
set -euo pipefail

# Phase-3-only ablation for the number of CDC-generated virtual Novel features.
#
# Reuses the ALREADY COMPLETED Stage-2 Phase-2 checkpoints:
#   exp/scannet200/stage2_phase2_k1_cdc
#   exp/scannet200/stage2_phase2_k5_cdc
#   exp/scannetpp/stage2_phase2_k1_cdc
#   exp/scannetpp/stage2_phase2_k5_cdc
#
# The ONLY method variable changed is:
#   model.virtual_samples = M
# with default sweep:
#   M in {2, 4, 8, 16, 32}
#
# All other Phase-3 settings are inherited from the original dataset/K config,
# including virtual_loss_weight=0.5, optimizer, epochs, evaluator and seeds.
# The original Phase-3 virtual radius is recovered from the existing baseline
# Phase-3 train.log when possible, so this remains a clean M-only ablation.
#
# By default, M=4 reuses the already completed baseline Phase-3 result rather
# than rerunning the same experiment. Set REUSE_BASELINE_M4=0 to rerun M=4.
#
# Outputs:
#   exp/<dataset>/Virtual_Samples_Ablation/k<K>/M<M>/...
#
# Examples:
#   GPU_LIST=0,3,5 bash scripts/run_phase3_virtual_samples_ablation.sh
#   DATASETS=scannet200 K_LIST=1 bash scripts/run_phase3_virtual_samples_ablation.sh
#   M_VALUES="2 8 16 32" REUSE_BASELINE_M4=0 bash scripts/run_phase3_virtual_samples_ablation.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,3,5}"
DATASETS="${DATASETS:-scannet200 scannetpp}"
K_LIST="${K_LIST:-1 5}"
M_VALUES="${M_VALUES:-2 4 8 16 32}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
REUSE_BASELINE_M4="${REUSE_BASELINE_M4:-1}"
PHASE3_RADIUS="${PHASE3_RADIUS:-auto}"
VIRTUAL_LOSS_WEIGHT="${VIRTUAL_LOSS_WEIGHT:-0.5}"

REGIS=(regis1 regis2 regis3 regis4 regis5)
SEEDS=(10 20 30 40 50)

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS=${#GPU_ARRAY[@]}
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "[ERROR] GPU_LIST is empty"
  exit 1
fi

DATASETS="${DATASETS//,/ }"
K_LIST="${K_LIST//,/ }"
M_VALUES="${M_VALUES//,/ }"
read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a K_ARRAY <<< "${K_LIST}"
read -r -a M_ARRAY <<< "${M_VALUES}"

for dataset in "${DATASET_ARRAY[@]}"; do
  if [ "${dataset}" != "scannet200" ] && [ "${dataset}" != "scannetpp" ]; then
    echo "[ERROR] DATASETS currently supports scannet200 and/or scannetpp; got ${dataset}"
    exit 1
  fi
done

for k in "${K_ARRAY[@]}"; do
  if [ "${k}" != "1" ] && [ "${k}" != "5" ]; then
    echo "[ERROR] K_LIST supports only 1 and/or 5; got ${k}"
    exit 1
  fi
done

for m in "${M_ARRAY[@]}"; do
  if ! [[ "${m}" =~ ^[0-9]+$ ]] || [ "${m}" -lt 1 ]; then
    echo "[ERROR] Every M value must be a positive integer; got ${m}"
    exit 1
  fi
done

launch_train() {
  local dataset="$1"
  local config="$2"
  local exp_name="$3"
  shift 3

  echo ""
  echo "============================================================"
  echo "[LAUNCH] ${dataset}/${config}"
  echo "Experiment : ./exp/${dataset}/${exp_name}"
  echo "GPUs       : ${GPU_LIST} (${NUM_GPUS})"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  sh scripts/train.sh \
    -p "${PYTHON_BIN}" \
    -d "${dataset}" \
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

# Recover the exact virtual radius used by the original M=4 baseline for this
# dataset/K/registration. If that cannot be parsed, fall back to the Stage-1
# RecommendedRef, then finally to the historical 0.025 default.
resolve_radius() {
  local dataset="$1"
  local k="$2"
  local regis="$3"

  if [ "${PHASE3_RADIUS}" != "auto" ]; then
    printf '%s' "${PHASE3_RADIUS}"
    return 0
  fi

  local baseline_log="./exp/${dataset}/stage2_phase3_k${k}_cdc/${regis}/train.log"
  local stage1_log="./exp/${dataset}/cdc_stage1_r24/train.log"

  "${PYTHON_BIN}" - "${baseline_log}" "${stage1_log}" <<'PY'
import os
import re
import sys

baseline_log, stage1_log = sys.argv[1], sys.argv[2]

patterns = [
    re.compile(r"virtual_radius\s*[=:]\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"['\"]virtual_radius['\"]\s*:\s*([0-9]*\.?[0-9]+)"),
]

if os.path.isfile(baseline_log):
    text = open(baseline_log, "r", encoding="utf-8", errors="ignore").read()
    values = []
    for pat in patterns:
        values.extend(pat.findall(text))
    if values:
        print(f"{float(values[-1]):.8f}", end="")
        raise SystemExit(0)

if os.path.isfile(stage1_log):
    text = open(stage1_log, "r", encoding="utf-8", errors="ignore").read()
    values = re.findall(
        r"RecommendedRef\(q=[0-9.]+\):\s*([0-9]*\.?[0-9]+)", text
    )
    if values:
        print(f"{float(values[-1]):.8f}", end="")
        raise SystemExit(0)

print("0.02500000", end="")
PY
}

copy_baseline_m4_metrics() {
  local dataset="$1"
  local k="$2"
  local regis="$3"
  local target_dir="$4"

  local source_dir="./exp/${dataset}/stage2_phase3_k${k}_cdc/${regis}"
  local source_metric="${source_dir}/phase3_metrics_final.json"
  if [ ! -f "${source_metric}" ]; then
    echo "[ERROR] M=4 baseline metric missing: ${source_metric}"
    return 1
  fi

  mkdir -p "${target_dir}"
  cp -f "${source_metric}" "${target_dir}/phase3_metrics_final.json"
  if [ -f "${source_dir}/phase3_val_history.json" ]; then
    cp -f "${source_dir}/phase3_val_history.json" "${target_dir}/phase3_val_history.json"
  fi
  printf '%s\n' "${source_dir}" > "${target_dir}/baseline_source.txt"
}

summarize_m() {
  local dataset="$1"
  local k="$2"
  local m="$3"
  local root="./exp/${dataset}/Virtual_Samples_Ablation/k${k}/M${m}"

  "${PYTHON_BIN}" - "${root}" "${dataset}" "${k}" "${m}" <<'PY'
import csv
import json
import os
import statistics
import sys

root, dataset, k, m = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
records = []
for i in range(1, 6):
    regis = f"regis{i}"
    path = os.path.join(root, regis, "phase3_metrics_final.json")
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    rec = dict(rec)
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
    "dataset": dataset,
    "k_shot": k,
    "virtual_samples": m,
    "phase2_source": f"exp/{dataset}/stage2_phase2_k{k}_cdc",
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

print(f"\n===== {dataset} K={k} virtual_samples M={m} =====")
for key in keys:
    a = aggregate[key]
    print(f"{key:22s}: {a['mean']:.6f} +/- {a['std']:.6f}")
print(f"JSON: {json_path}")
print(f"CSV : {csv_path}")
print("============================================================")
PY
}

summarize_sweep() {
  local dataset="$1"
  local k="$2"
  local root="./exp/${dataset}/Virtual_Samples_Ablation/k${k}"
  local m_values="$3"

  "${PYTHON_BIN}" - "${root}" "${dataset}" "${k}" "${m_values}" <<'PY'
import csv
import json
import os
import sys

root, dataset, k, m_values = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
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
    json.dump({"dataset": dataset, "k_shot": k, "rows": rows}, f, indent=2)
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

print(f"[SUMMARY] {dataset} K={k}: {csv_path}")
PY
}

run_dataset_k_m() {
  local dataset="$1"
  local k="$2"
  local m="$3"
  local config="semseg-pt-v3m1-0-stage2-phase3-k${k}-cdc-aug"
  local config_path="configs/${dataset}/${config}.py"
  local p2_root="./exp/${dataset}/stage2_phase2_k${k}_cdc"
  local target_root="./exp/${dataset}/Virtual_Samples_Ablation/k${k}/M${m}"

  if [ ! -f "${config_path}" ]; then
    echo "[ERROR] Missing Phase-3 config: ${config_path}"
    exit 1
  fi
  if [ ! -d "${p2_root}" ]; then
    echo "[ERROR] Missing Phase-2 root: ${p2_root}"
    exit 1
  fi

  echo ""
  echo "####################################################################"
  echo "# ${dataset} K=${k}: virtual_samples M=${m}"
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
      echo "[ERROR] Missing Phase-2 checkpoint: ${p2_ckpt}"
      exit 1
    fi

    if [ "${m}" = "4" ] && [ "${REUSE_BASELINE_M4}" = "1" ]; then
      if [ -f "${expected}" ] && [ "${REUSE_COMPLETED}" = "1" ]; then
        echo "[REUSE] copied M=4 baseline ${dataset} K=${k} ${regis}"
      else
        copy_baseline_m4_metrics "${dataset}" "${k}" "${regis}" "${out_dir}"
        echo "[BASELINE] reused original M=4 result for ${dataset} K=${k} ${regis}"
      fi
      continue
    fi

    if check_output_dir "${out_dir}" "${expected}" "${dataset} K=${k} M=${m} ${regis}"; then
      continue
    else
      local status=$?
      if [ "${status}" -eq 2 ]; then exit 1; fi
    fi

    local radius
    radius="$(resolve_radius "${dataset}" "${k}" "${regis}")"
    echo "[RADIUS] ${dataset} K=${k} ${regis}: ${radius}"

    local exp_name="Virtual_Samples_Ablation/k${k}/M${m}/${regis}"
    launch_train \
      "${dataset}" \
      "${config}" \
      "${exp_name}" \
      -o "phase2_weight='${p2_ckpt}'" \
      -o "phase3_regis_name='${regis}'" \
      -o "data.train.k_shot=${k}" \
      -o "data.train.seed=${seed}" \
      -o "model.virtual_samples=${m}" \
      -o "model.virtual_radius=${radius}" \
      -o "model.virtual_loss_weight=${VIRTUAL_LOSS_WEIGHT}"

    if [ ! -f "${expected}" ]; then
      echo "[ERROR] Missing Phase-3 metric after training: ${expected}"
      exit 1
    fi
    if [ ! -f "${out_dir}/model/model_best.pth" ]; then
      echo "[ERROR] Missing best checkpoint: ${out_dir}/model/model_best.pth"
      exit 1
    fi
  done

  summarize_m "${dataset}" "${k}" "${m}"
}

echo "============================================================"
echo "Phase-3 virtual-sample-count ablation"
echo "Datasets : ${DATASETS}"
echo "K list   : ${K_LIST}"
echo "M values : ${M_VALUES}"
echo "GPUs     : ${GPU_LIST} (${NUM_GPUS})"
echo "M=4 reuse: ${REUSE_BASELINE_M4}"
echo "============================================================"

for dataset in "${DATASET_ARRAY[@]}"; do
  for k in "${K_ARRAY[@]}"; do
    for m in "${M_ARRAY[@]}"; do
      run_dataset_k_m "${dataset}" "${k}" "${m}"
    done
    summarize_sweep "${dataset}" "${k}" "${M_VALUES}"
  done
done

echo ""
echo "[DONE] Phase-3 virtual-sample-count ablation completed."
