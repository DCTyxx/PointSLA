#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline for the NEW protocol:
#   official GFS-VL task-adapted checkpoint (pres_exp/<tag>_k<K>)
#       -> merged Stage-1 + old Phase-2 geometry learning
#          * CDC tangent: Base + trusted K-shot Novel GT
#          * semantic projector: Base GT only
#          * safe boundary/radius: Base only
#       -> Phase-3 CDC Novel feature augmentation
#
# Official model-zoo tags:
#   ScanNet    : sc_k1 / sc_k5
#   ScanNet200 : sc200_k1 / sc200_k5
#   ScanNet++  : scpp_k1 / scpp_k5
#
# The official task checkpoint already contains the Phase-1 GFS-VL adaptation,
# therefore the default protocol does NOT rerun GFS-VL registration training.
# pres_exp/backbone_weights and pres_exp/3d_vlm_weight are kept as official
# source assets for reproducibility, but are not consumed by this direct-checkpoint
# path. This avoids changing the official task representation before geometry.

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,3,5}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-./pres_exp}"
BACKBONE_ROOT="${BACKBONE_ROOT:-${PRETRAIN_ROOT}/backbone_weights}"
VLM_ROOT="${VLM_ROOT:-${PRETRAIN_ROOT}/3d_vlm_weight}"
DATASETS="${DATASETS:-scannet}"
K_LIST="${K_LIST:-1 5}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
PHASE3_RADIUS="${PHASE3_RADIUS:-auto}"

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
if [ "${DATASETS}" = "all" ]; then
  DATASETS="scannet scannet200 scannetpp"
fi
read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a K_ARRAY <<< "${K_LIST}"

for k in "${K_ARRAY[@]}"; do
  if [ "${k}" != "1" ] && [ "${k}" != "5" ]; then
    echo "[ERROR] K_LIST supports only 1 and/or 5; got ${k}"
    exit 1
  fi
done

if [ ! -d "${PRETRAIN_ROOT}" ]; then
  echo "[ERROR] pres_exp root not found: ${PRETRAIN_ROOT}"
  exit 1
fi

if [ -d "${BACKBONE_ROOT}" ]; then
  echo "[INFO] Official PTv3 backbone assets: ${BACKBONE_ROOT}"
else
  echo "[WARN] Official backbone directory not found: ${BACKBONE_ROOT}"
fi
if [ -d "${VLM_ROOT}" ]; then
  echo "[INFO] Official 3D-VLM assets: ${VLM_ROOT}"
else
  echo "[WARN] Official 3D-VLM directory not found: ${VLM_ROOT}"
fi

dataset_tag() {
  case "$1" in
    scannet)    echo "sc" ;;
    scannet200) echo "sc200" ;;
    scannetpp)  echo "scpp" ;;
    *) echo "[ERROR] Unsupported dataset: $1" >&2; return 1 ;;
  esac
}

resolve_task_ckpt() {
  local dataset="$1"
  local k="$2"
  local regis="$3"
  local tag
  tag="$(dataset_tag "${dataset}")"
  local root="${PRETRAIN_ROOT}/${tag}_k${k}"

  if [ ! -d "${root}" ]; then
    echo "[ERROR] Official model-zoo folder not found: ${root}" >&2
    return 1
  fi

  local candidates=(
    "${root}/model/${regis}_model_last.pth"
    "${root}/${regis}_model_last.pth"
    "${root}/model/${regis}_model_best.pth"
    "${root}/${regis}_model_best.pth"
  )
  local path
  for path in "${candidates[@]}"; do
    if [ -f "${path}" ]; then
      printf '%s' "${path}"
      return 0
    fi
  done

  path="$(find "${root}" -type f \( -name "${regis}_model_last.pth" -o -name "${regis}_model_best.pth" \) | sort | head -n 1 || true)"
  if [ -n "${path}" ] && [ -f "${path}" ]; then
    printf '%s' "${path}"
    return 0
  fi

  echo "[ERROR] Cannot find official ${regis} checkpoint under ${root}" >&2
  return 1
}

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
    echo "[ERROR] Existing/incomplete output: ${out_dir}"
    echo "        Expected completion file: ${expected}"
    echo "        Remove it or set ALLOW_EXISTING=1 intentionally."
    return 2
  fi
  return 1
}

resolve_radius() {
  local merged_dir="$1"
  if [ "${PHASE3_RADIUS}" != "auto" ]; then
    printf '%s' "${PHASE3_RADIUS}"
    return 0
  fi

  local log_file="${merged_dir}/train.log"
  if [ ! -f "${log_file}" ]; then
    echo "[ERROR] Cannot resolve Base-only safe radius: missing ${log_file}" >&2
    return 1
  fi

  local radius
  radius="$(grep -oE 'RecommendedRef\(q=[0-9.]+\):[[:space:]]*[0-9.]+' "${log_file}" | tail -n 1 | awk '{print $NF}' || true)"
  if [ -z "${radius}" ]; then
    echo "[ERROR] No RecommendedRef found in ${log_file}" >&2
    return 1
  fi

  "${PYTHON_BIN}" - "${radius}" <<'PY'
import sys
x = float(sys.argv[1])
if x <= 0:
    raise SystemExit(f"Base-only RecommendedRef must be > 0, got {x}")
print(f"{x:.8f}", end="")
PY
}

summarize_phase3() {
  local dataset="$1"
  local k="$2"
  local root="./exp/${dataset}/merged_phase3_k${k}"

  "${PYTHON_BIN}" - "${root}" "${dataset}" "${k}" <<'PY'
import csv
import json
import os
import statistics
import sys

root, dataset, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
records = []
for i in range(1, 6):
    regis = f"regis{i}"
    metric_path = os.path.join(root, regis, "phase3_metrics_final.json")
    radius_path = os.path.join(root, regis, "base_only_radius.txt")
    with open(metric_path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    with open(radius_path, "r", encoding="utf-8") as f:
        radius = float(f.read().strip())
    rec = dict(rec)
    rec["base_only_virtual_radius"] = radius
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
radii = [r["base_only_virtual_radius"] for r in records]
aggregate["base_only_virtual_radius"] = {
    "mean": statistics.mean(radii),
    "std": statistics.pstdev(radii),
    "min": min(radii),
    "max": max(radii),
}

payload = {
    "dataset": dataset,
    "protocol": "official GFS-VL task checkpoint -> merged geometry -> Phase3",
    "safe_boundary": "Base-only",
    "k_shot": k,
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
    fields = ["regis", "base_only_virtual_radius"] + keys
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for rec in records:
        writer.writerow({key: rec[key] for key in fields})

print(f"\n===== {dataset} merged geometry K={k}, 5-regis summary =====")
for key in keys:
    a = aggregate[key]
    print(f"{key:22s}: {a['mean']:.6f} +/- {a['std']:.6f}")
a = aggregate["base_only_virtual_radius"]
print(f"{'Base-only radius':22s}: {a['mean']:.6f} +/- {a['std']:.6f}")
print(f"JSON: {json_path}")
print(f"CSV : {csv_path}")
print("============================================================")
PY
}

run_dataset_k() {
  local dataset="$1"
  local k="$2"
  local merged_config="semseg-pt-v3m1-0-stage2-merged-geometry-k${k}"
  local phase3_config="semseg-pt-v3m1-0-stage2-phase3-k${k}-cdc-aug"

  for cfg in "${merged_config}" "${phase3_config}"; do
    if [ ! -f "configs/${dataset}/${cfg}.py" ]; then
      echo "[ERROR] Missing config: configs/${dataset}/${cfg}.py"
      exit 1
    fi
  done

  echo ""
  echo "####################################################################"
  echo "# DATASET=${dataset} K=${k}"
  echo "# merged geometry: Base+Novel tangent; Base-only semantic boundary"
  echo "####################################################################"

  local i
  for i in "${!REGIS[@]}"; do
    local regis="${REGIS[$i]}"
    local seed="${SEEDS[$i]}"
    local task_ckpt
    task_ckpt="$(resolve_task_ckpt "${dataset}" "${k}" "${regis}")"
    echo "[FOUND] official ${dataset} K=${k} ${regis}: ${task_ckpt}"

    local merged_exp="merged_stage1_phase2_k${k}/${regis}"
    local merged_dir="./exp/${dataset}/${merged_exp}"
    local merged_ckpt="${merged_dir}/model/model_last.pth"

    if check_output_dir "${merged_dir}" "${merged_ckpt}" "Merged K=${k} ${regis}"; then
      :
    else
      local status=$?
      if [ "${status}" -eq 2 ]; then exit 1; fi
      launch_train \
        "${dataset}" \
        "${merged_config}" \
        "${merged_exp}" \
        -o "task_weight='${task_ckpt}'" \
        -o "merged_regis_name='${regis}'" \
        -o "data.train.k_shot=${k}" \
        -o "data.train.seed=${seed}"
    fi

    if [ ! -f "${merged_ckpt}" ]; then
      echo "[ERROR] Missing merged geometry checkpoint: ${merged_ckpt}"
      exit 1
    fi

    local radius
    radius="$(resolve_radius "${merged_dir}")"
    echo "[BASE-ONLY RADIUS] ${dataset} K=${k} ${regis}: ${radius}"

    local phase3_exp="merged_phase3_k${k}/${regis}"
    local phase3_dir="./exp/${dataset}/${phase3_exp}"
    local expected="${phase3_dir}/phase3_metrics_final.json"

    if check_output_dir "${phase3_dir}" "${expected}" "Phase3 K=${k} ${regis}"; then
      :
    else
      local status=$?
      if [ "${status}" -eq 2 ]; then exit 1; fi
      launch_train \
        "${dataset}" \
        "${phase3_config}" \
        "${phase3_exp}" \
        -o "phase2_weight='${merged_ckpt}'" \
        -o "phase3_regis_name='${regis}'" \
        -o "data.train.k_shot=${k}" \
        -o "data.train.seed=${seed}" \
        -o "model.virtual_radius=${radius}"
    fi

    if [ ! -f "${expected}" ]; then
      echo "[ERROR] Missing Phase3 metrics: ${expected}"
      exit 1
    fi
    if [ ! -f "${phase3_dir}/model/model_best.pth" ]; then
      echo "[ERROR] Missing Phase3 best checkpoint: ${phase3_dir}/model/model_best.pth"
      exit 1
    fi
    printf '%s\n' "${radius}" > "${phase3_dir}/base_only_radius.txt"
  done

  summarize_phase3 "${dataset}" "${k}"
}

echo "============================================================"
echo "Merged Stage1 + Stage2-Phase2 pipeline"
echo "pres_exp   : ${PRETRAIN_ROOT}"
echo "datasets   : ${DATASETS}"
echo "K list     : ${K_LIST}"
echo "GPUs/run   : ${GPU_LIST} (${NUM_GPUS})"
echo "radius     : ${PHASE3_RADIUS} (auto = per-regis Base-only P05)"
echo "============================================================"

for dataset in "${DATASET_ARRAY[@]}"; do
  dataset_tag "${dataset}" >/dev/null
  for k in "${K_ARRAY[@]}"; do
    run_dataset_k "${dataset}" "${k}"
  done
done

echo ""
echo "[DONE] merged geometry pipeline completed."
