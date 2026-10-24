#!/usr/bin/env bash
set -euo pipefail

# ScanNet++ K=1 Phase-3 direction ablation with a best-so-far feature cache.
#
# For every direction mode, five registrations are trained/evaluated normally.
# Full train/val feature pairs are NOT exported for every registration. After
# precise evaluation, Stage2Phase3CachedBestFeaturePairExporter compares the
# current precise HM with the mode-level cache and exports only if the current
# registration is better than the cached best.
#
# At any time there is at most one feature snapshot per mode:
#   exp/scannetpp/Direction_Ablation_k1/<mode>/best_feature_pairs/
#
# Modes:
#   real_only       : real-feature fine-tuning only
#   isotropic       : random full-space directions
#   random_subspace : fixed random rank-r orthonormal subspace
#   local_pca       : same-class local PCA subspace
#   cdc             : our learned LSL/CDC directions
#
# Usage:
#   GPU_LIST=0,3,5 bash scripts/run_scannetpp_k1_direction_ablation.sh
#   MODES="cdc" GPU_LIST=0,3,5 bash scripts/run_scannetpp_k1_direction_ablation.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,3,5}"
MODES="${MODES:-real_only isotropic random_subspace local_pca cdc}"
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
    real_only|isotropic|random_subspace|local_pca|cdc) ;;
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
  shift
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
    "Cannot recover split Phase-3/Base-P05 radius. Set PHASE3_RADIUS explicitly."
)
PY
}

run_mode() {
  local mode="$1"
  local mode_root="${ABLATION_ROOT}/${mode}"
  mkdir -p "${mode_root}"

  echo ""
  echo "####################################################################"
  echo "# ScanNet++ K=1 direction ablation: ${mode}"
  echo "# Shared feature cache: ${mode_root}/best_feature_pairs"
  echo "####################################################################"

  local i
  for i in "${!REGIS[@]}"; do
    local regis="${REGIS[$i]}"
    local seed="${SEEDS[$i]}"
    local p2_ckpt="${PHASE2_ROOT}/${regis}/model/model_last.pth"
    local out_dir="${mode_root}/${regis}"
    local metrics="${out_dir}/phase3_metrics_final.json"
    local cache_status="${out_dir}/feature_export_cache_status.json"

    if [ ! -f "${p2_ckpt}" ]; then
      echo "[ERROR] Missing split Phase-2 checkpoint: ${p2_ckpt}"
      exit 1
    fi

    if [ -f "${metrics}" ] && [ -f "${cache_status}" ] && [ "${REUSE_COMPLETED}" = "1" ]; then
      echo "[REUSE] ${mode}/${regis}: metrics + cache status found"
      continue
    fi

    # Old runs may have precise metrics but no cache status because they were
    # produced before the cache implementation. We intentionally rerun those.
    if [ -d "${out_dir}" ] && [ ! -f "${metrics}" ] && [ "${ALLOW_EXISTING}" != "1" ]; then
      echo "[ERROR] Existing incomplete output: ${out_dir}"
      echo "        Remove it or set ALLOW_EXISTING=1 intentionally."
      exit 1
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

    if [ ! -f "${metrics}" ]; then
      echo "[ERROR] Missing Phase-3 metrics: ${metrics}"
      exit 1
    fi
    if [ ! -f "${cache_status}" ]; then
      echo "[ERROR] Missing feature-cache status: ${cache_status}"
      exit 1
    fi

    printf '%s\n' "${radius}" > "${out_dir}/base_only_radius.txt"
    printf '%s\n' "${mode}" > "${out_dir}/augmentation_mode.txt"
  done

  summarize_mode "${mode}"
}

summarize_mode() {
  local mode="$1"
  local root="${ABLATION_ROOT}/${mode}"

  "${PYTHON_BIN}" - "${root}" "${mode}" <<'PY'
import csv
import json
import math
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

best = max(records, key=lambda r: float(r["hm_mIoU"]))
best_regis = best["regis"]
best_hm = float(best["hm_mIoU"])
cache_root = os.path.join(root, "best_feature_pairs")
state_path = os.path.join(cache_root, "cache_state.json")
if not os.path.isfile(state_path):
    raise RuntimeError(f"Feature cache state missing: {state_path}")
with open(state_path, "r", encoding="utf-8") as f:
    state = json.load(f)

if state.get("best_regis") != best_regis:
    raise RuntimeError(
        f"Feature cache mismatch for {mode}: cache={state.get('best_regis')} "
        f"but precise-HM best={best_regis}"
    )
if not math.isclose(float(state["best_metric"]), best_hm, rel_tol=0.0, abs_tol=1e-10):
    raise RuntimeError(
        f"Feature cache metric mismatch for {mode}: cache={state['best_metric']} "
        f"but precise-HM best={best_hm}"
    )
for split in ("train", "val"):
    split_root = os.path.join(cache_root, split)
    required = [
        "features.npy", "labels.npy", "scene_ids.npy", "point_ids.npy",
        "virtual_features.npy", "directions.npy", "virtual_anchor_indices.npy",
    ]
    missing = [x for x in required if not os.path.isfile(os.path.join(split_root, x))]
    if missing:
        raise RuntimeError(f"Incomplete feature cache {mode}/{split}: missing={missing}")

payload = {
    "dataset": "scannetpp",
    "k_shot": 1,
    "direction_mode": mode,
    "phase2_source": "exp/scannetpp/split_phase2_k1",
    "protocol": "fixed split Phase2 -> Novel-head-only Phase3 direction ablation",
    "num_regis": 5,
    "std_definition": "population_std_ddof0",
    "best_regis": best_regis,
    "best_feature_pairs": cache_root,
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
print(f"Best regis by precise HM: {best_regis} ({best_hm:.6f})")
print(f"Feature cache           : {cache_root}")
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
    row = {"mode": mode, "best_regis": payload["best_regis"]}
    for key in keys:
        row[f"{key}_mean"] = payload["aggregate"][key]["mean"]
        row[f"{key}_std"] = payload["aggregate"][key]["std"]
    rows.append(row)

json_path = os.path.join(root, "direction_ablation_summary.json")
csv_path = os.path.join(root, "direction_ablation_summary.csv")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"dataset": "scannetpp", "k_shot": 1, "rows": rows}, f, indent=2)
fields = ["mode", "best_regis"]
for key in keys:
    fields.extend([f"{key}_mean", f"{key}_std"])
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"[SUMMARY] {csv_path}")
PY
}

echo "============================================================"
echo "ScanNet++ K=1 direction ablation + best-so-far feature cache"
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
echo "[DONE] ScanNet++ K=1 direction ablation + cached best feature export completed."
