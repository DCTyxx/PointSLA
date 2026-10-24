#!/usr/bin/env bash
set -euo pipefail

# ScanNet++ K=1 Phase-3 direction ablation + optimized final feature export.
#
# Feature-export policy per mode:
#   1) train/evaluate regis1..regis5 normally;
#   2) no feature export for regis1..regis4;
#   3) after regis5 precise evaluation, select the FINAL best regis by HM;
#   4) load that regis model_best and export validation features ONCE;
#   5) save trusted Novel anchors only (Base and label=-1 are omitted);
#   6) write one-pass uncompressed NPZ shards.
#
# Final output:
#   exp/scannetpp/Direction_Ablation_k1/<mode>/best_feature_pairs/
#     cache_state.json
#     best_regis.json
#     val/
#       manifest.json
#       metadata.json
#       scene_names.json
#       shards/shard_*.npz
#
# Each shard contains:
#   features          float32 [A,D]       Novel real anchors only
#   labels            int64   [A]
#   scene_ids         int32   [A]
#   point_ids         int32   [A]
#   virtual_features  float32 [M*A,D]     empty for real_only
#   directions        float32 [M*A,D]     empty for real_only
#
# Virtual rows are anchor-major, so within a shard:
#   virtual_anchor_index = arange(A).repeat_interleave(M)
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
EXPORT_VERSION="final_best_val_novel_sharded_v2"
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

  local explicit="${ABLATION_ROOT}/__MODE__/${regis}/base_only_radius.txt"
  # Caller may replace __MODE__ before using this candidate; baseline/stage1 are
  # the authoritative fallbacks and work for both fresh and reused runs.
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

patterns = [
    re.compile(r"virtual_radius\s*[=:]\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"['\"]virtual_radius['\"]\s*:\s*([0-9]*\.?[0-9]+)"),
]
if os.path.isfile(baseline_log):
    text = open(baseline_log, "r", encoding="utf-8", errors="ignore").read()
    vals = []
    for pat in patterns:
        vals.extend(pat.findall(text))
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

has_final_export() {
  local mode_root="$1"
  local state="${mode_root}/best_feature_pairs/cache_state.json"
  local manifest="${mode_root}/best_feature_pairs/val/manifest.json"
  if [ ! -f "${state}" ] || [ ! -f "${manifest}" ]; then
    return 1
  fi
  "${PYTHON_BIN}" - "${state}" "${EXPORT_VERSION}" <<'PY'
import json
import sys
path, expected = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    state = json.load(f)
raise SystemExit(0 if state.get("export_version") == expected else 1)
PY
}

summarize_mode() {
  local mode="$1"
  local root="${ABLATION_ROOT}/${mode}"

  "${PYTHON_BIN}" - "${root}" "${mode}" "${EXPORT_VERSION}" <<'PY'
import csv
import json
import math
import os
import statistics
import sys

root, mode, export_version = sys.argv[1:4]
records = []
for i in range(1, 6):
    regis = f"regis{i}"
    path = os.path.join(root, regis, "phase3_metrics_final.json")
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing precise metrics: {path}")
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
feature_root = os.path.join(root, "best_feature_pairs")
state_path = os.path.join(feature_root, "cache_state.json")
manifest_path = os.path.join(feature_root, "val", "manifest.json")
meta_path = os.path.join(feature_root, "val", "metadata.json")
for path in (state_path, manifest_path, meta_path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Final feature export incomplete: missing {path}")

with open(state_path, "r", encoding="utf-8") as f:
    state = json.load(f)
if state.get("export_version") != export_version:
    raise RuntimeError(
        f"Old feature-export format for {mode}: {state.get('export_version')} != {export_version}"
    )
if state.get("best_regis") != best_regis:
    raise RuntimeError(
        f"Final feature mismatch for {mode}: export={state.get('best_regis')} "
        f"but precise-HM best={best_regis}"
    )
if not math.isclose(float(state["best_metric"]), best_hm, rel_tol=0.0, abs_tol=1e-10):
    raise RuntimeError(
        f"Final feature metric mismatch for {mode}: export={state['best_metric']} "
        f"but precise-HM best={best_hm}"
    )

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
if meta.get("features_scope") != "trusted Novel validation anchors only; Base and label=-1 omitted":
    raise RuntimeError(f"Unexpected feature scope for {mode}: {meta.get('features_scope')}")
if not bool(meta.get("single_pass", False)):
    raise RuntimeError(f"Feature export for {mode} is not marked single-pass")

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)
for row in manifest:
    path = os.path.join(feature_root, "val", row["file"])
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing feature shard: {path}")

payload = {
    "dataset": "scannetpp",
    "k_shot": 1,
    "direction_mode": mode,
    "phase2_source": "exp/scannetpp/split_phase2_k1",
    "protocol": "fixed split Phase2 -> Novel-head-only Phase3 direction ablation",
    "num_regis": 5,
    "std_definition": "population_std_ddof0",
    "best_regis": best_regis,
    "best_feature_pairs": feature_root,
    "feature_export_version": export_version,
    "feature_export_scope": "validation Novel anchors only",
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
print(f"Best regis by precise HM : {best_regis} ({best_hm:.6f})")
print(f"Novel val anchors        : {meta['num_novel_anchors']}")
print(f"Virtual features         : {meta['num_virtual_features']}")
print(f"Feature shards           : {meta['num_shards']}")
print(f"Final feature snapshot   : {feature_root}")
print("============================================================")
PY
}

run_mode() {
  local mode="$1"
  local mode_root="${ABLATION_ROOT}/${mode}"
  mkdir -p "${mode_root}"

  echo ""
  echo "####################################################################"
  echo "# ScanNet++ K=1 direction ablation: ${mode}"
  echo "# Final feature output: ${mode_root}/best_feature_pairs/val"
  echo "####################################################################"

  local i
  for i in "${!REGIS[@]}"; do
    local regis="${REGIS[$i]}"
    local seed="${SEEDS[$i]}"
    local p2_ckpt="${PHASE2_ROOT}/${regis}/model/model_last.pth"
    local out_dir="${mode_root}/${regis}"
    local metrics="${out_dir}/phase3_metrics_final.json"

    if [ ! -f "${p2_ckpt}" ]; then
      echo "[ERROR] Missing split Phase-2 checkpoint: ${p2_ckpt}"
      exit 1
    fi

    # For regis1~4, precise metrics are sufficient for reuse: these runs never
    # need feature export in the optimized protocol.
    if [ "${regis}" != "regis5" ] && [ -f "${metrics}" ] && [ "${REUSE_COMPLETED}" = "1" ]; then
      radius="$(resolve_radius "${regis}")"
      printf '%s\n' "${radius}" > "${out_dir}/base_only_radius.txt"
      echo "[REUSE] ${mode}/${regis}: precise metrics found; no feature export needed"
      continue
    fi

    # regis5 can be reused only when the NEW final-best export already exists.
    if [ "${regis}" = "regis5" ] && [ -f "${metrics}" ] && \
       [ "${REUSE_COMPLETED}" = "1" ] && has_final_export "${mode_root}"; then
      radius="$(resolve_radius "${regis}")"
      printf '%s\n' "${radius}" > "${out_dir}/base_only_radius.txt"
      echo "[REUSE] ${mode}/regis5 + optimized final feature export"
      continue
    fi

    if [ -d "${out_dir}" ] && [ ! -f "${metrics}" ] && [ "${ALLOW_EXISTING}" != "1" ]; then
      echo "[ERROR] Existing incomplete output: ${out_dir}"
      echo "        Remove it or set ALLOW_EXISTING=1 intentionally."
      exit 1
    fi

    # An old completed regis5 without the v2 export is intentionally rerun so
    # the final hook can select the best among the already-existing five metrics
    # and export validation features exactly once.
    if [ "${regis}" = "regis5" ] && [ -f "${metrics}" ] && ! has_final_export "${mode_root}"; then
      echo "[RERUN] ${mode}/regis5: precise metrics exist but optimized final export is missing"
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
    printf '%s\n' "${radius}" > "${out_dir}/base_only_radius.txt"
    printf '%s\n' "${mode}" > "${out_dir}/augmentation_mode.txt"
  done

  if ! has_final_export "${mode_root}"; then
    echo "[ERROR] Optimized final feature export missing for mode=${mode}"
    exit 1
  fi
  summarize_mode "${mode}"
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
echo "ScanNet++ K=1 direction ablation + FINAL-best val-only export"
echo "Phase2 root : ${PHASE2_ROOT}"
echo "Modes       : ${MODES}"
echo "GPUs/run    : ${GPU_LIST} (${NUM_GPUS})"
echo "Virtual M   : 4"
echo "Radius      : ${PHASE3_RADIUS} (auto = split Base-only P05)"
echo "PCA KNN     : ${PCA_KNN}"
echo "PCA chunk   : ${PCA_CHUNK_SIZE}"
echo "Export      : val only, Novel anchors only, single-pass shards"
echo "============================================================"

for mode in "${MODE_ARRAY[@]}"; do
  run_mode "${mode}"
done
summarize_all

echo ""
echo "[DONE] ScanNet++ K=1 direction ablation + optimized feature export completed."
