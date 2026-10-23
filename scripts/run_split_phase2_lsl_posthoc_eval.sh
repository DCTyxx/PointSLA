#!/usr/bin/env bash
set -euo pipefail

# Post-hoc LSL analysis for EXISTING split Phase-2 checkpoints.
# No parameter is updated. For each regis, report five metrics for
# Overall / Base / Novel:
#   TAS, Intra-EVR, Inter-EVR, GDG, Mean Principal Angle (degrees).
#
# Default examples:
#   DATASETS=scannet200 K_LIST="1 5" GPU=0 bash scripts/run_split_phase2_lsl_posthoc_eval.sh
#   DATASETS="scannet scannet200 scannetpp" K_LIST="1 5" GPU=0 bash scripts/run_split_phase2_lsl_posthoc_eval.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-scannet200}"
K_LIST="${K_LIST:-1 5}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-6}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"

REGIS=(regis1 regis2 regis3 regis4 regis5)
SEEDS=(10 20 30 40 50)

DATASETS="${DATASETS//,/ }"
K_LIST="${K_LIST//,/ }"
read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a K_ARRAY <<< "${K_LIST}"

for dataset in "${DATASET_ARRAY[@]}"; do
  case "${dataset}" in
    scannet|scannet200|scannetpp) ;;
    *) echo "[ERROR] Unsupported dataset: ${dataset}"; exit 1 ;;
  esac

done

for k in "${K_ARRAY[@]}"; do
  if [ "${k}" != "1" ] && [ "${k}" != "5" ]; then
    echo "[ERROR] K_LIST supports only 1 and/or 5; got ${k}"
    exit 1
  fi
done

summarize_k() {
  local dataset="$1"
  local k="$2"
  local root="./exp/${dataset}/split_phase2_k${k}"

  "${PYTHON_BIN}" - "${root}" "${dataset}" "${k}" <<'PY'
import csv
import json
import os
import statistics
import sys

root, dataset, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
fields = ["TAS", "Intra-EVR", "Inter-EVR", "GDG", "Mean-Principal-Angle-Deg"]
groups = ["overall", "base", "novel"]

per_regis = []
for i in range(1, 6):
    regis = f"regis{i}"
    path = os.path.join(root, regis, "phase2_lsl_posthoc.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    per_regis.append({"regis": regis, "groups": payload["groups"]})

aggregate = {}
for group in groups:
    aggregate[group] = {}
    for field in fields:
        vals = [float(r["groups"][group][field]) for r in per_regis]
        aggregate[group][field] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals),
            "min": min(vals),
            "max": max(vals),
        }

summary = {
    "dataset": dataset,
    "k_shot": k,
    "num_regis": 5,
    "std_definition": "population_std_ddof0",
    "aggregation_within_regis": "anchor_weighted",
    "per_regis": per_regis,
    "aggregate": aggregate,
}

json_path = os.path.join(root, "phase2_lsl_metrics_summary.json")
csv_path = os.path.join(root, "phase2_lsl_metrics_summary.csv")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

rows = []
for group in groups:
    row = {"group": group}
    for field in fields:
        safe = field.lower().replace("-", "_")
        row[f"{safe}_mean"] = aggregate[group][field]["mean"]
        row[f"{safe}_std"] = aggregate[group][field]["std"]
    rows.append(row)

csv_fields = ["group"]
for field in fields:
    safe = field.lower().replace("-", "_")
    csv_fields += [f"{safe}_mean", f"{safe}_std"]
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n===== {dataset} K={k} LSL summary over 5 registrations =====")
for group in groups:
    print(group.upper())
    for field in fields:
        a = aggregate[group][field]
        print(f"  {field:28s}: {a['mean']:.6f} +/- {a['std']:.6f}")
print(f"JSON: {json_path}")
print(f"CSV : {csv_path}")
print("============================================================")
PY
}

for dataset in "${DATASET_ARRAY[@]}"; do
  for k in "${K_ARRAY[@]}"; do
    config="configs/${dataset}/semseg-pt-v3m1-0-stage2-phase2-k${k}-cdc-adapt.py"
    root="./exp/${dataset}/split_phase2_k${k}"

    if [ ! -f "${config}" ]; then
      echo "[ERROR] Missing config: ${config}"
      exit 1
    fi
    if [ ! -d "${root}" ]; then
      echo "[ERROR] Missing split Phase-2 root: ${root}"
      exit 1
    fi

    echo ""
    echo "####################################################################"
    echo "# ${dataset} K=${k} post-hoc LSL analysis"
    echo "####################################################################"

    for i in "${!REGIS[@]}"; do
      regis="${REGIS[$i]}"
      seed="${SEEDS[$i]}"
      ckpt="${root}/${regis}/model/model_last.pth"
      output="${root}/${regis}/phase2_lsl_posthoc.json"

      if [ ! -f "${ckpt}" ]; then
        echo "[ERROR] Missing Phase-2 checkpoint: ${ckpt}"
        exit 1
      fi
      if [ -f "${output}" ] && [ "${REUSE_COMPLETED}" = "1" ]; then
        echo "[REUSE] ${output}"
        continue
      fi

      echo "[EVAL] ${dataset} K=${k} ${regis} seed=${seed}"
      CUDA_VISIBLE_DEVICES="${GPU}" \
      "${PYTHON_BIN}" tools/eval_phase2_lsl_metrics.py \
        --config "${config}" \
        --checkpoint "${ckpt}" \
        --output "${output}" \
        --k-shot "${k}" \
        --seed "${seed}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --device cuda:0
    done

    summarize_k "${dataset}" "${k}"
  done
done
