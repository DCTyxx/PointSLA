#!/usr/bin/env bash
set -euo pipefail

# One-click ScanNet200 CDC pipeline:
#   Stage 1: Base-only CDC geometry learning (shared by K=1 and K=5)
#   Stage 2 Phase 1: independently adapt PTv3 + Base/Novel heads per registration
#                    while freezing the Stage-1 CDC stack
#   Stage 2 Phase 2: freeze task model and adapt CDC per registration
#   Stage 2 Phase 3: freeze PTv3/CDC/Base head and refine Novel head with
#                    CDC-generated virtual Novel features
#
# K=1 and K=5 both use regis1..regis5 with seeds 10,20,30,40,50.
# Every registration is launched in a fresh process, so no optimizer/model state
# leaks from one support split into another.

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,3,5}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
PHASE3_RADIUS="${PHASE3_RADIUS:-auto}"

DATASET="scannet200"
BASE_CKPT="${BASE_CKPT:-./exp/scannet200/ptv3_bk/model_best.pth}"
STAGE1_RUN="${STAGE1_RUN:-cdc_stage1_r24}"
STAGE1_CONFIG="semseg-pt-v3m1-0-conservative-boundary-semantic-tangent-cdcbase-frozen-r24"

REGIS=(regis1 regis2 regis3 regis4 regis5)
SEEDS=(10 20 30 40 50)

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS=${#GPU_ARRAY[@]}
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "[ERROR] GPU_LIST is empty"
  exit 1
fi

if [ ! -f "${BASE_CKPT}" ]; then
  echo "[ERROR] ScanNet200 Base checkpoint not found: ${BASE_CKPT}"
  exit 1
fi

for cfg in \
  "${STAGE1_CONFIG}" \
  "semseg-pt-v3m1-0-stage2-phase1-k1-frozen-cdc" \
  "semseg-pt-v3m1-0-stage2-phase1-k5-frozen-cdc" \
  "semseg-pt-v3m1-0-stage2-phase2-k1-cdc-adapt" \
  "semseg-pt-v3m1-0-stage2-phase2-k5-cdc-adapt" \
  "semseg-pt-v3m1-0-stage2-phase3-k1-cdc-aug" \
  "semseg-pt-v3m1-0-stage2-phase3-k5-cdc-aug"
do
  path="configs/${DATASET}/${cfg}.py"
  if [ ! -f "${path}" ]; then
    echo "[ERROR] Missing config: ${path}"
    exit 1
  fi
done

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
    echo "[ERROR] Incomplete/existing output directory: ${out_dir}"
    echo "        Expected completion file: ${expected}"
    echo "        Remove the directory or set ALLOW_EXISTING=1 intentionally."
    return 2
  fi
  return 1
}

# -----------------------------------------------------------------------------
# Stage 1: shared Base-only CDC geometry prior
# -----------------------------------------------------------------------------
STAGE1_DIR="./exp/${DATASET}/${STAGE1_RUN}"
STAGE1_CKPT="${STAGE1_DIR}/model/model_best.pth"

if check_output_dir "${STAGE1_DIR}" "${STAGE1_CKPT}" "Stage-1"; then
  :
else
  status=$?
  if [ "${status}" -eq 2 ]; then exit 1; fi
  launch_train \
    "${STAGE1_CONFIG}" \
    "${STAGE1_RUN}" \
    -o "weight='${BASE_CKPT}'"
fi

if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "[ERROR] Stage-1 finished without model_best.pth: ${STAGE1_CKPT}"
  exit 1
fi

echo "[READY] Stage-1 checkpoint: ${STAGE1_CKPT}"

# Use the ScanNet200 Stage-1 conservative P05 reference for Phase-3 augmentation
# when available, rather than blindly reusing ScanNet's 0.025.
if [ "${PHASE3_RADIUS}" = "auto" ]; then
  LOG_FILE="${STAGE1_DIR}/train.log"
  if [ -f "${LOG_FILE}" ]; then
    RESOLVED_RADIUS="$(grep -oE 'RecommendedRef\(q=[0-9.]+\):[[:space:]]*[0-9.]+' "${LOG_FILE}" | tail -n 1 | awk '{print $NF}' || true)"
  else
    RESOLVED_RADIUS=""
  fi
  if [ -z "${RESOLVED_RADIUS}" ]; then
    RESOLVED_RADIUS="0.025"
    echo "[WARN] Could not parse Stage-1 RecommendedRef; fallback radius=${RESOLVED_RADIUS}"
  else
    echo "[AUTO] Phase-3 virtual radius from ScanNet200 Stage-1: ${RESOLVED_RADIUS}"
  fi
else
  RESOLVED_RADIUS="${PHASE3_RADIUS}"
  echo "[USER] Phase-3 virtual radius: ${RESOLVED_RADIUS}"
fi

run_phase1() {
  local k="$1"
  local config="semseg-pt-v3m1-0-stage2-phase1-k${k}-frozen-cdc"
  local root="stage2_phase1_k${k}_cdc"

  echo ""
  echo "################ Stage-2 Phase-1 K=${k} ################"
  for i in "${!REGIS[@]}"; do
    local regis="${REGIS[$i]}"
    local seed="${SEEDS[$i]}"
    local exp_name="${root}/${regis}"
    local out_dir="./exp/${DATASET}/${exp_name}"
    local expected="${out_dir}/model/${regis}_model_last.pth"

    if check_output_dir "${out_dir}" "${expected}" "Phase1 K=${k} ${regis}"; then
      continue
    else
      local status=$?
      if [ "${status}" -eq 2 ]; then exit 1; fi
    fi

    # Although the config defaults to all five registrations, the full-pipeline
    # launcher deliberately runs one registration per fresh process. This resets
    # model, optimizer, scheduler, and Novel-head state for every support split.
    launch_train \
      "${config}" \
      "${exp_name}" \
      -o "backbone_weight='${STAGE1_CKPT}'" \
      -o "regis_train_list=['${regis}']"

    if [ ! -f "${expected}" ]; then
      echo "[ERROR] Missing Phase-1 checkpoint: ${expected}"
      exit 1
    fi
    echo "[DONE] Phase1 K=${k} ${regis} seed=${seed}: ${expected}"
  done
}

run_phase2() {
  local k="$1"
  local config="semseg-pt-v3m1-0-stage2-phase2-k${k}-cdc-adapt"
  local p1_root="stage2_phase1_k${k}_cdc"
  local root="stage2_phase2_k${k}_cdc"

  echo ""
  echo "################ Stage-2 Phase-2 K=${k} ################"
  for i in "${!REGIS[@]}"; do
    local regis="${REGIS[$i]}"
    local seed="${SEEDS[$i]}"
    local p1_ckpt="./exp/${DATASET}/${p1_root}/${regis}/model/${regis}_model_last.pth"
    local exp_name="${root}/${regis}"
    local out_dir="./exp/${DATASET}/${exp_name}"
    local expected="${out_dir}/model/model_last.pth"

    if [ ! -f "${p1_ckpt}" ]; then
      echo "[ERROR] Missing matching Phase-1 checkpoint: ${p1_ckpt}"
      exit 1
    fi

    if check_output_dir "${out_dir}" "${expected}" "Phase2 K=${k} ${regis}"; then
      continue
    else
      local status=$?
      if [ "${status}" -eq 2 ]; then exit 1; fi
    fi

    launch_train \
      "${config}" \
      "${exp_name}" \
      -o "phase1_weight='${p1_ckpt}'" \
      -o "phase2_regis_name='${regis}'" \
      -o "data.train.k_shot=${k}" \
      -o "data.train.seed=${seed}"

    if [ ! -f "${expected}" ]; then
      echo "[ERROR] Missing Phase-2 checkpoint: ${expected}"
      exit 1
    fi
    echo "[DONE] Phase2 K=${k} ${regis}: ${expected}"
  done
}

run_phase3() {
  local k="$1"
  local config="semseg-pt-v3m1-0-stage2-phase3-k${k}-cdc-aug"
  local p2_root="stage2_phase2_k${k}_cdc"
  local root="stage2_phase3_k${k}_cdc"

  echo ""
  echo "################ Stage-2 Phase-3 K=${k} ################"
  for i in "${!REGIS[@]}"; do
    local regis="${REGIS[$i]}"
    local seed="${SEEDS[$i]}"
    local p2_ckpt="./exp/${DATASET}/${p2_root}/${regis}/model/model_last.pth"
    local exp_name="${root}/${regis}"
    local out_dir="./exp/${DATASET}/${exp_name}"
    local expected="${out_dir}/phase3_metrics_final.json"

    if [ ! -f "${p2_ckpt}" ]; then
      echo "[ERROR] Missing matching Phase-2 checkpoint: ${p2_ckpt}"
      exit 1
    fi

    if check_output_dir "${out_dir}" "${expected}" "Phase3 K=${k} ${regis}"; then
      continue
    else
      local status=$?
      if [ "${status}" -eq 2 ]; then exit 1; fi
    fi

    launch_train \
      "${config}" \
      "${exp_name}" \
      -o "phase2_weight='${p2_ckpt}'" \
      -o "phase3_regis_name='${regis}'" \
      -o "data.train.k_shot=${k}" \
      -o "data.train.seed=${seed}" \
      -o "model.virtual_radius=${RESOLVED_RADIUS}"

    if [ ! -f "${expected}" ]; then
      echo "[ERROR] Missing Phase-3 final metrics: ${expected}"
      exit 1
    fi
    if [ ! -f "${out_dir}/model/model_best.pth" ]; then
      echo "[ERROR] Missing Phase-3 best-HM checkpoint: ${out_dir}/model/model_best.pth"
      exit 1
    fi
    echo "[DONE] Phase3 K=${k} ${regis}: ${expected}"
  done
}

summarize_phase3() {
  local k="$1"
  local root="./exp/${DATASET}/stage2_phase3_k${k}_cdc"

  "${PYTHON_BIN}" - "${root}" "${k}" <<'PY'
import csv
import json
import os
import statistics
import sys

root, k = sys.argv[1], int(sys.argv[2])
records = []
for i in range(1, 6):
    regis = f"regis{i}"
    path = os.path.join(root, regis, "phase3_metrics_final.json")
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    if rec.get("regis") != regis:
        raise RuntimeError(f"Expected {regis}, got {rec.get('regis')} in {path}")
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
    writer = csv.DictWriter(f, fieldnames=["regis"] + keys)
    writer.writeheader()
    for rec in records:
        writer.writerow({"regis": rec["regis"], **{key: rec[key] for key in keys}})
    writer.writerow({"regis": "mean", **{key: aggregate[key]["mean"] for key in keys}})
    writer.writerow({"regis": "std", **{key: aggregate[key]["std"] for key in keys}})

print(f"\n============= ScanNet200 Phase-3 K={k}, 5-regis summary =============")
for key in keys:
    x = aggregate[key]
    print(f"{key:22s}: {x['mean']:.6f} +/- {x['std']:.6f}")
print(f"JSON: {json_path}")
print(f"CSV : {csv_path}")
print("====================================================================")
PY
}

# Shared Stage 1 is complete; now run both support regimes end-to-end.
for K in 1 5; do
  run_phase1 "${K}"
  run_phase2 "${K}"
  run_phase3 "${K}"
  summarize_phase3 "${K}"
done

echo ""
echo "===================================================================="
echo "ScanNet200 CDC pipeline completed: Stage1 + K1/K5 Phase1/2/3"
echo "Stage-1: ${STAGE1_CKPT}"
echo "Phase-3 radius: ${RESOLVED_RADIUS}"
echo "K=1 summary: ./exp/${DATASET}/stage2_phase3_k1_cdc/phase3_metrics_summary.json"
echo "K=5 summary: ./exp/${DATASET}/stage2_phase3_k5_cdc/phase3_metrics_summary.json"
echo "===================================================================="
