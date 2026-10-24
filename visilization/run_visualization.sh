#!/usr/bin/env bash
set -euo pipefail

# PointSLA / GFS-VL qualitative visualization pipeline.
#
# Default protocol:
#   - datasets: ScanNet, ScanNet200, ScanNet++
#   - K=1
#   - choose Our best registration by precise HM under exp/<dataset>/split_phase3_k1
#   - evaluate official GFS-VL baseline using the SAME registration
#   - evaluate Our Phase-3 best checkpoint
#   - compare scene-level mIoU and select 20 scenes with largest Our-Baseline gain
#   - export paper-ready x y z label r g b point tables
#
# Usage:
#   GPU=4 bash visilization/run_visualization.sh
#   DATASETS="scannetpp" GPU=4 bash visilization/run_visualization.sh
#   K=5 TOPK=20 GPU=4 bash visilization/run_visualization.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-scannet scannet200 scannetpp}"
K="${K:-1}"
TOPK="${TOPK:-20}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-./pres_exp}"
VIS_ROOT="${VIS_ROOT:-./visilization}"
REUSE_EVAL="${REUSE_EVAL:-1}"

if [ "${K}" != "1" ] && [ "${K}" != "5" ]; then
  echo "[ERROR] K must be 1 or 5, got ${K}"
  exit 1
fi

DATASETS="${DATASETS//,/ }"
read -r -a DATASET_ARRAY <<< "${DATASETS}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

baseline_tag() {
  case "$1" in
    scannet)    echo "sc" ;;
    scannet200) echo "sc200" ;;
    scannetpp)  echo "scpp" ;;
    *) echo "[ERROR] Unsupported dataset: $1" >&2; return 1 ;;
  esac
}

data_root() {
  case "$1" in
    scannet|scannet200) echo "./data/ScanNet200" ;;
    scannetpp)          echo "./data/ScanNetpp" ;;
    *) echo "[ERROR] Unsupported dataset: $1" >&2; return 1 ;;
  esac
}

resolve_pair() {
  local dataset="$1"
  local tag="$2"
  local our_root="./exp/${dataset}/split_phase3_k${K}"
  local baseline_root="${PRETRAIN_ROOT}/${tag}_k${K}"

  "${PYTHON_BIN}" - "${our_root}" "${baseline_root}" <<'PY'
import json
import os
import sys

our_root, baseline_root = sys.argv[1:3]
records = []
for i in range(1, 6):
    regis = f"regis{i}"
    path = os.path.join(our_root, regis, "phase3_metrics_final.json")
    if not os.path.isfile(path):
        raise SystemExit(f"Missing Our precise metric: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    records.append((float(rec["hm_mIoU"]), regis))

best_hm, regis = max(records, key=lambda x: x[0])
our_ckpt = os.path.join(our_root, regis, "model", "model_best.pth")
if not os.path.isfile(our_ckpt):
    raise SystemExit(f"Missing Our best checkpoint: {our_ckpt}")

# Match the exact resolver preference used by the split task-space pipeline.
candidates = [
    os.path.join(baseline_root, "model", f"{regis}_model_last.pth"),
    os.path.join(baseline_root, f"{regis}_model_last.pth"),
    os.path.join(baseline_root, "model", f"{regis}_model_best.pth"),
    os.path.join(baseline_root, f"{regis}_model_best.pth"),
]
baseline_ckpt = next((p for p in candidates if os.path.isfile(p)), None)
if baseline_ckpt is None:
    for root, _, files in os.walk(baseline_root):
        for name in sorted(files):
            if name in {f"{regis}_model_last.pth", f"{regis}_model_best.pth"}:
                baseline_ckpt = os.path.join(root, name)
                break
        if baseline_ckpt:
            break
if baseline_ckpt is None:
    raise SystemExit(
        f"Cannot find official baseline checkpoint for {regis} under {baseline_root}"
    )

print(f"{regis}\t{best_hm:.12f}\t{baseline_ckpt}\t{our_ckpt}")
PY
}

has_scene_outputs() {
  local vis_dir="$1"
  find "${vis_dir}" -mindepth 2 -maxdepth 2 -name query_pred.npy -print -quit 2>/dev/null | grep -q .
}

run_eval() {
  local dataset="$1"
  local role="$2"
  local config="$3"
  local ckpt="$4"
  local out_dir="$5"
  local vis_dir="$6"

  local marker="${out_dir}/checkpoint.txt"
  if [ "${REUSE_EVAL}" = "1" ] && [ -f "${marker}" ] && [ -d "${vis_dir}" ]; then
    local old_ckpt
    old_ckpt="$(cat "${marker}")"
    if [ "${old_ckpt}" = "${ckpt}" ] && has_scene_outputs "${vis_dir}"; then
      echo "[REUSE] ${dataset}/${role}: ${ckpt}"
      return 0
    fi
  fi

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}" "${vis_dir}"

  echo ""
  echo "============================================================"
  echo "[EVAL] ${dataset} ${role}"
  echo "Config : configs/${dataset}/${config}.py"
  echo "CKPT   : ${ckpt}"
  echo "Output : ${out_dir}"
  echo "Vis    : ${vis_dir}"
  echo "GPU    : ${GPU}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  "${PYTHON_BIN}" -u visilization/test_gfs_vis.py \
    --config-file "configs/${dataset}/${config}.py" \
    --num-gpus 1 \
    --options \
      save_path="${out_dir}" \
      weight="${ckpt}" \
      vis="${vis_dir}" \
      empty_cache=True

  if ! has_scene_outputs "${vis_dir}"; then
    echo "[ERROR] Evaluation produced no scene visualization outputs: ${vis_dir}"
    exit 1
  fi
  if [ ! -f "${out_dir}/evaluation_metrics.json" ]; then
    echo "[ERROR] Evaluation metrics were not saved: ${out_dir}/evaluation_metrics.json"
    exit 1
  fi
  printf '%s\n' "${ckpt}" > "${marker}"
}

for dataset in "${DATASET_ARRAY[@]}"; do
  tag="$(baseline_tag "${dataset}")"
  root="$(data_root "${dataset}")"
  if [ ! -d "${root}" ]; then
    echo "[ERROR] Dataset root not found: ${root}"
    exit 1
  fi

  pair="$(resolve_pair "${dataset}" "${tag}")"
  IFS=$'\t' read -r regis best_hm baseline_ckpt our_ckpt <<< "${pair}"

  echo ""
  echo "####################################################################"
  echo "# ${dataset} K=${K} visualization"
  echo "# matched regis : ${regis}"
  echo "# Our precise HM: ${best_hm}"
  echo "# Baseline CKPT : ${baseline_ckpt}"
  echo "# Our CKPT      : ${our_ckpt}"
  echo "####################################################################"

  eval_root="${VIS_ROOT}/evaluation/${dataset}"
  baseline_out="${eval_root}/baseline"
  ours_out="${eval_root}/ours"
  baseline_vis="${baseline_out}/scenes"
  ours_vis="${ours_out}/scenes"

  baseline_cfg="semseg-pt-v3m1-0-gfsregistrain_k${K}"
  ours_cfg="semseg-pt-v3m1-0-stage2-phase3-k${K}-cdc-aug"

  run_eval "${dataset}" "baseline" "${baseline_cfg}" "${baseline_ckpt}" "${baseline_out}" "${baseline_vis}"
  run_eval "${dataset}" "ours" "${ours_cfg}" "${our_ckpt}" "${ours_out}" "${ours_vis}"

  comparison_dir="${VIS_ROOT}/comparison/${dataset}"
  showcase_dir="${VIS_ROOT}/showcase/${dataset}"
  rm -rf "${comparison_dir}"
  mkdir -p "${comparison_dir}"

  "${PYTHON_BIN}" visilization/build_showcase.py \
    --dataset "${dataset}" \
    --baseline-vis "${baseline_vis}" \
    --ours-vis "${ours_vis}" \
    --data-root "${root}" \
    --comparison-dir "${comparison_dir}" \
    --showcase-dir "${showcase_dir}" \
    --topk "${TOPK}" \
    --regis "${regis}" \
    --k-shot "${K}" \
    --baseline-ckpt "${baseline_ckpt}" \
    --ours-ckpt "${our_ckpt}"

done

echo ""
echo "===================================================================="
echo "Visualization pipeline completed."
echo "Evaluation : ${VIS_ROOT}/evaluation/"
echo "Comparison : ${VIS_ROOT}/comparison/"
echo "Showcase   : ${VIS_ROOT}/showcase/"
echo "===================================================================="
