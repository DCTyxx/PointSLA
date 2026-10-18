#!/usr/bin/env bash
set -euo pipefail

# One-click ScanNet hyperparameter ablation pipeline.
#
# Default study (K=1, five independent registration splits per setting):
#   r           : 8, 16, 24, 32
#   lambda_disc : 0, 0.05, 0.1, 0.2
#   rho         : 0.01, 0.025, 0.05, 0.075, 0.10
#   M           : 1, 2, 4, 8
#   lambda_v    : 0, 0.25, 0.5, 1.0
#
# Every value gets its own COMPLETE and isolated pipeline:
#   Stage-1 -> Phase-1 x5 -> Phase-2 x5 -> Phase-3 x5 -> summary
#
# Outputs are intentionally rooted directly under:
#   ./exp/Ablation_Studies/<hyperparameter>/<value>/
# rather than ./exp/scannet/.  scripts/train.sh always prefixes ./exp/scannet,
# so experiment names use ../Ablation_Studies/... to resolve to the requested
# top-level ablation directory without modifying the repository-wide train.sh.
#
# Defaults correspond to the current full method:
#   r=24, lambda_disc=0.1, rho=0.025, M=4, lambda_v=0.5
#
# Environment examples:
#   GPU_LIST=0,3,5 bash scripts/run_scannet_hparam_ablation_full.sh
#   K_LIST="1 5" GPU_LIST=0,3,5 bash scripts/run_scannet_hparam_ablation_full.sh
#   ABLATIONS="rho M" bash scripts/run_scannet_hparam_ablation_full.sh
#   RHO_VALUES="0.01 0.025 0.05" bash scripts/run_scannet_hparam_ablation_full.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,3,5}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"

# Default: the sensitivity study is run on ScanNet K=1, as planned in the paper.
# Set K_LIST="1 5" to repeat every setting for both shot protocols.
K_LIST="${K_LIST:-1}"
ABLATIONS="${ABLATIONS:-r lambda_disc rho M lambda_v}"

R_VALUES="${R_VALUES:-8 16 24 32}"
LAMBDA_DISC_VALUES="${LAMBDA_DISC_VALUES:-0 0.05 0.1 0.2}"
RHO_VALUES="${RHO_VALUES:-0.01 0.025 0.05 0.075 0.10}"
M_VALUES="${M_VALUES:-1 2 4 8}"
LAMBDA_V_VALUES="${LAMBDA_V_VALUES:-0 0.25 0.5 1.0}"

# Full-method defaults.  A single ablated hyperparameter replaces exactly one
# of these while all remaining method settings stay fixed.
BASE_R="${BASE_R:-24}"
BASE_LAMBDA_DISC="${BASE_LAMBDA_DISC:-0.1}"
BASE_RHO="${BASE_RHO:-0.025}"
BASE_M="${BASE_M:-4}"
BASE_LAMBDA_V="${BASE_LAMBDA_V:-0.5}"

DATASET="scannet"
BASE_CKPT="${BASE_CKPT:-./exp/scannet/ptv3_bk/model/model_best.pth}"
VLM_CKPT="${VLM_CKPT:-./pretrain_weights/sparseunet32_636.pth}"
DATA_ROOT="${DATA_ROOT:-./data/ScanNet200}"
ABLATION_ROOT="./exp/Ablation_Studies"

STAGE1_CONFIG="semseg-pt-v3m1-0-conservative-boundary-semantic-tangent-cdcbase-frozen-r24"
P1_K1_CONFIG="semseg-pt-v3m1-0-stage2-phase1-k1-frozen-cdc"
P1_K5_CONFIG="semseg-pt-v3m1-0-stage2-phase1-k5-frozen-cdc"
P2_K1_CONFIG="semseg-pt-v3m1-0-stage2-phase2-k1-cdc-adapt"
P2_K5_CONFIG="semseg-pt-v3m1-0-stage2-phase2-k5-cdc-adapt"
P3_K1_CONFIG="semseg-pt-v3m1-0-stage2-phase3-k1-cdc-aug"
P3_K5_CONFIG="semseg-pt-v3m1-0-stage2-phase3-k5-cdc-aug"

REGIS=(regis1 regis2 regis3 regis4 regis5)
SEEDS=(10 20 30 40 50)

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS=${#GPU_ARRAY[@]}
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "[ERROR] GPU_LIST is empty"
  exit 1
fi

# Accept comma-separated or whitespace-separated user overrides.
K_LIST="${K_LIST//,/ }"
ABLATIONS="${ABLATIONS//,/ }"
read -r -a K_VALUES <<< "${K_LIST}"
read -r -a ABLATION_NAMES <<< "${ABLATIONS}"

mkdir -p "${ABLATION_ROOT}"

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
if [ ! -f "${BASE_CKPT}" ]; then
  echo "[ERROR] ScanNet Base checkpoint not found: ${BASE_CKPT}"
  exit 1
fi
if [ ! -f "${VLM_CKPT}" ]; then
  echo "[ERROR] GFS-VL 3D-VLM checkpoint not found: ${VLM_CKPT}"
  echo "        Phase-1 requires sparseunet32_636.pth."
  exit 1
fi
if [ ! -d "${DATA_ROOT}" ]; then
  echo "[ERROR] ScanNet data root not found: ${DATA_ROOT}"
  exit 1
fi

for cfg in \
  "${STAGE1_CONFIG}" \
  "${P1_K1_CONFIG}" "${P1_K5_CONFIG}" \
  "${P2_K1_CONFIG}" "${P2_K5_CONFIG}" \
  "${P3_K1_CONFIG}" "${P3_K5_CONFIG}"
do
  path="configs/${DATASET}/${cfg}.py"
  if [ ! -f "${path}" ]; then
    echo "[ERROR] Missing config: ${path}"
    exit 1
  fi
done

for k in "${K_VALUES[@]}"; do
  if [ "${k}" != "1" ] && [ "${k}" != "5" ]; then
    echo "[ERROR] K_LIST supports only 1 and/or 5; got: ${k}"
    exit 1
  fi
done

values_for_param() {
  case "$1" in
    r)           echo "${R_VALUES}" ;;
    lambda_disc) echo "${LAMBDA_DISC_VALUES}" ;;
    rho)         echo "${RHO_VALUES}" ;;
    M)           echo "${M_VALUES}" ;;
    lambda_v)    echo "${LAMBDA_V_VALUES}" ;;
    *)
      echo "[ERROR] Unknown ablation parameter: $1" >&2
      return 1
      ;;
  esac
}

for param in "${ABLATION_NAMES[@]}"; do
  values_for_param "${param}" >/dev/null
  mkdir -p "${ABLATION_ROOT}/${param}"
done

launch_train() {
  local config="$1"
  local exp_name="$2"
  shift 2

  echo ""
  echo "============================================================"
  echo "[LAUNCH] ${DATASET}/${config}"
  echo "Experiment : ./exp/scannet/${exp_name}"
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
    echo "        Remove it or set ALLOW_EXISTING=1 intentionally."
    return 2
  fi
  return 1
}

write_setting_meta() {
  local root="$1"
  local param="$2"
  local value="$3"
  local rank="$4"
  local lambda_disc="$5"
  local rho="$6"
  local m="$7"
  local lambda_v="$8"

  "${PYTHON_BIN}" - "${root}" "${param}" "${value}" "${rank}" "${lambda_disc}" "${rho}" "${m}" "${lambda_v}" "${K_LIST}" "${GLOBAL_SEED}" <<'PY'
import json
import os
import sys

root, param, value, rank, ld, rho, m, lv, k_list, seed = sys.argv[1:]
os.makedirs(root, exist_ok=True)
payload = {
    "dataset": "scannet",
    "ablation_parameter": param,
    "ablation_value": value,
    "full_method_hparams": {
        "r": int(rank),
        "lambda_disc": float(ld),
        "rho": float(rho),
        "M": int(m),
        "lambda_v": float(lv),
    },
    "k_list": [int(x) for x in k_list.replace(',', ' ').split()],
    "registration_seeds": [10, 20, 30, 40, 50],
    "global_seed": int(seed),
    "protocol": "full isolated Stage1 -> Phase1 -> Phase2 -> Phase3 for every value",
}
with open(os.path.join(root, "ablation_meta.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY
}

summarize_phase3() {
  local setting_root="$1"
  local k="$2"
  local param="$3"
  local value="$4"
  local rank="$5"
  local lambda_disc="$6"
  local rho="$7"
  local m="$8"
  local lambda_v="$9"

  local root="${setting_root}/k${k}/phase3"

  "${PYTHON_BIN}" - "${root}" "${k}" "${param}" "${value}" "${rank}" "${lambda_disc}" "${rho}" "${m}" "${lambda_v}" <<'PY'
import csv
import json
import os
import statistics
import sys

root = sys.argv[1]
k = int(sys.argv[2])
param, value = sys.argv[3], sys.argv[4]
rank = int(sys.argv[5])
ld = float(sys.argv[6])
rho = float(sys.argv[7])
m = int(sys.argv[8])
lv = float(sys.argv[9])

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
    "dataset": "scannet",
    "ablation_parameter": param,
    "ablation_value": value,
    "k_shot": k,
    "num_regis": 5,
    "std_definition": "population_std_ddof0",
    "hparams": {
        "r": rank,
        "lambda_disc": ld,
        "rho": rho,
        "M": m,
        "lambda_v": lv,
    },
    "per_regis": records,
    "aggregate": aggregate,
}
json_path = os.path.join(os.path.dirname(root), "phase3_metrics_summary.json")
csv_path = os.path.join(os.path.dirname(root), "phase3_metrics_summary.csv")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["regis"] + keys)
    writer.writeheader()
    for rec in records:
        writer.writerow({"regis": rec["regis"], **{key: rec[key] for key in keys}})
    writer.writerow({"regis": "mean", **{key: aggregate[key]["mean"] for key in keys}})
    writer.writerow({"regis": "std", **{key: aggregate[key]["std"] for key in keys}})

print(f"\n===== {param}={value} | ScanNet Phase-3 K={k}, 5-regis summary =====")
for key in keys:
    x = aggregate[key]
    print(f"{key:22s}: {x['mean']:.6f} +/- {x['std']:.6f}")
print(f"JSON: {json_path}")
print(f"CSV : {csv_path}")
print("====================================================================")
PY
}

run_setting() {
  local param="$1"
  local value="$2"

  local rank="${BASE_R}"
  local lambda_disc="${BASE_LAMBDA_DISC}"
  local rho="${BASE_RHO}"
  local m="${BASE_M}"
  local lambda_v="${BASE_LAMBDA_V}"

  case "${param}" in
    r)           rank="${value}" ;;
    lambda_disc) lambda_disc="${value}" ;;
    rho)         rho="${value}" ;;
    M)           m="${value}" ;;
    lambda_v)    lambda_v="${value}" ;;
    *) echo "[ERROR] Unknown parameter: ${param}"; exit 1 ;;
  esac

  if [ "${rank}" -gt 32 ]; then
    echo "[ERROR] Current tangent.knn=32 requires r<=32; got r=${rank}"
    exit 1
  fi
  if [ "${rank}" -lt 1 ]; then
    echo "[ERROR] r must be positive; got ${rank}"
    exit 1
  fi
  if [ "${m}" -lt 1 ]; then
    echo "[ERROR] M must be >=1; got ${m}"
    exit 1
  fi

  local setting_root="${ABLATION_ROOT}/${param}/${value}"
  # train.sh prefixes ./exp/scannet; ../Ablation_Studies resolves to ./exp/Ablation_Studies.
  local exp_prefix="../Ablation_Studies/${param}/${value}"
  mkdir -p "${setting_root}"

  write_setting_meta "${setting_root}" "${param}" "${value}" \
    "${rank}" "${lambda_disc}" "${rho}" "${m}" "${lambda_v}"

  echo ""
  echo "####################################################################"
  echo "# ABLATION: ${param}=${value}"
  echo "# r=${rank}, lambda_disc=${lambda_disc}, rho=${rho}, M=${m}, lambda_v=${lambda_v}"
  echo "# output: ${setting_root}"
  echo "####################################################################"

  # ---------------------------------------------------------------------------
  # Stage 1: full Base-only geometry pretraining for THIS ablation value.
  # ---------------------------------------------------------------------------
  local stage1_dir="${setting_root}/stage1"
  local stage1_ckpt="${stage1_dir}/model/model_best.pth"

  if check_output_dir "${stage1_dir}" "${stage1_ckpt}" "${param}=${value} Stage-1"; then
    :
  else
    local status=$?
    if [ "${status}" -eq 2 ]; then exit 1; fi
    launch_train \
      "${STAGE1_CONFIG}" \
      "${exp_prefix}/stage1" \
      -o "weight='${BASE_CKPT}'" \
      -o "seed=${GLOBAL_SEED}" \
      -o "model.cdc.rank=${rank}" \
      -o "model.tangent.disc_loss_weight=${lambda_disc}"
  fi

  if [ ! -f "${stage1_ckpt}" ]; then
    echo "[ERROR] Stage-1 finished without model_best.pth: ${stage1_ckpt}"
    exit 1
  fi

  # ---------------------------------------------------------------------------
  # K-shot pipelines.  Each regis is a fresh process.
  # ---------------------------------------------------------------------------
  local k
  for k in "${K_VALUES[@]}"; do
    local p1_config p2_config p3_config
    if [ "${k}" = "1" ]; then
      p1_config="${P1_K1_CONFIG}"
      p2_config="${P2_K1_CONFIG}"
      p3_config="${P3_K1_CONFIG}"
    else
      p1_config="${P1_K5_CONFIG}"
      p2_config="${P2_K5_CONFIG}"
      p3_config="${P3_K5_CONFIG}"
    fi

    echo ""
    echo "################ ${param}=${value} | K=${k} Phase-1 ################"
    local i
    for i in "${!REGIS[@]}"; do
      local regis="${REGIS[$i]}"
      local seed="${SEEDS[$i]}"
      local p1_dir="${setting_root}/k${k}/phase1/${regis}"
      local p1_ckpt="${p1_dir}/model/${regis}_model_last.pth"

      if check_output_dir "${p1_dir}" "${p1_ckpt}" "${param}=${value} K=${k} Phase1 ${regis}"; then
        continue
      else
        local status=$?
        if [ "${status}" -eq 2 ]; then exit 1; fi
      fi

      launch_train \
        "${p1_config}" \
        "${exp_prefix}/k${k}/phase1/${regis}" \
        -o "seed=${GLOBAL_SEED}" \
        -o "backbone_weight='${stage1_ckpt}'" \
        -o "vlm_3d_weight='${VLM_CKPT}'" \
        -o "regis_train_list=['${regis}']" \
        -o "model.cdc.rank=${rank}" \
        -o "model.tangent.disc_loss_weight=${lambda_disc}"

      if [ ! -f "${p1_ckpt}" ]; then
        echo "[ERROR] Missing Phase-1 checkpoint: ${p1_ckpt}"
        exit 1
      fi
      echo "[DONE] ${param}=${value} K=${k} Phase1 ${regis} seed=${seed}"
    done

    echo ""
    echo "################ ${param}=${value} | K=${k} Phase-2 ################"
    for i in "${!REGIS[@]}"; do
      local regis="${REGIS[$i]}"
      local seed="${SEEDS[$i]}"
      local p1_ckpt="${setting_root}/k${k}/phase1/${regis}/model/${regis}_model_last.pth"
      local p2_dir="${setting_root}/k${k}/phase2/${regis}"
      local p2_ckpt="${p2_dir}/model/model_last.pth"

      if [ ! -f "${p1_ckpt}" ]; then
        echo "[ERROR] Missing matching Phase-1 checkpoint: ${p1_ckpt}"
        exit 1
      fi

      if check_output_dir "${p2_dir}" "${p2_ckpt}" "${param}=${value} K=${k} Phase2 ${regis}"; then
        continue
      else
        local status=$?
        if [ "${status}" -eq 2 ]; then exit 1; fi
      fi

      launch_train \
        "${p2_config}" \
        "${exp_prefix}/k${k}/phase2/${regis}" \
        -o "seed=${GLOBAL_SEED}" \
        -o "phase1_weight='${p1_ckpt}'" \
        -o "phase2_regis_name='${regis}'" \
        -o "data.train.k_shot=${k}" \
        -o "data.train.seed=${seed}" \
        -o "model.cdc.rank=${rank}" \
        -o "model.tangent.disc_loss_weight=${lambda_disc}"

      if [ ! -f "${p2_ckpt}" ]; then
        echo "[ERROR] Missing Phase-2 checkpoint: ${p2_ckpt}"
        exit 1
      fi
      echo "[DONE] ${param}=${value} K=${k} Phase2 ${regis}"
    done

    echo ""
    echo "################ ${param}=${value} | K=${k} Phase-3 ################"
    for i in "${!REGIS[@]}"; do
      local regis="${REGIS[$i]}"
      local seed="${SEEDS[$i]}"
      local p2_ckpt="${setting_root}/k${k}/phase2/${regis}/model/model_last.pth"
      local p3_dir="${setting_root}/k${k}/phase3/${regis}"
      local expected="${p3_dir}/phase3_metrics_final.json"

      if [ ! -f "${p2_ckpt}" ]; then
        echo "[ERROR] Missing matching Phase-2 checkpoint: ${p2_ckpt}"
        exit 1
      fi

      if check_output_dir "${p3_dir}" "${expected}" "${param}=${value} K=${k} Phase3 ${regis}"; then
        continue
      else
        local status=$?
        if [ "${status}" -eq 2 ]; then exit 1; fi
      fi

      launch_train \
        "${p3_config}" \
        "${exp_prefix}/k${k}/phase3/${regis}" \
        -o "seed=${GLOBAL_SEED}" \
        -o "phase2_weight='${p2_ckpt}'" \
        -o "phase3_regis_name='${regis}'" \
        -o "data.train.k_shot=${k}" \
        -o "data.train.seed=${seed}" \
        -o "model.cdc.rank=${rank}" \
        -o "model.tangent.disc_loss_weight=${lambda_disc}" \
        -o "model.virtual_radius=${rho}" \
        -o "model.virtual_samples=${m}" \
        -o "model.virtual_loss_weight=${lambda_v}"

      if [ ! -f "${expected}" ]; then
        echo "[ERROR] Missing Phase-3 final metrics: ${expected}"
        exit 1
      fi
      if [ ! -f "${p3_dir}/model/model_best.pth" ]; then
        echo "[ERROR] Missing Phase-3 best-HM checkpoint: ${p3_dir}/model/model_best.pth"
        exit 1
      fi
      echo "[DONE] ${param}=${value} K=${k} Phase3 ${regis}"
    done

    summarize_phase3 "${setting_root}" "${k}" "${param}" "${value}" \
      "${rank}" "${lambda_disc}" "${rho}" "${m}" "${lambda_v}"
  done
}

summarize_parameter() {
  local param="$1"
  local value_string
  value_string="$(values_for_param "${param}")"
  local param_root="${ABLATION_ROOT}/${param}"

  "${PYTHON_BIN}" - "${param_root}" "${param}" "${value_string}" "${K_LIST}" <<'PY'
import csv
import json
import os
import sys

root, param, value_string, k_string = sys.argv[1:]
values = value_string.split()
ks = [int(x) for x in k_string.replace(',', ' ').split()]
rows = []
for value in values:
    for k in ks:
        path = os.path.join(root, value, f"k{k}", "phase3_metrics_summary.json")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        agg = payload["aggregate"]
        row = {
            "hyperparameter": param,
            "value": value,
            "k_shot": k,
            "mIoU_mean": agg["mIoU"]["mean"],
            "mIoU_std": agg["mIoU"]["std"],
            "base_mIoU_mean": agg["base_mIoU"]["mean"],
            "base_mIoU_std": agg["base_mIoU"]["std"],
            "novel_mIoU_mean": agg["novel_mIoU"]["mean"],
            "novel_mIoU_std": agg["novel_mIoU"]["std"],
            "hm_mIoU_mean": agg["hm_mIoU"]["mean"],
            "hm_mIoU_std": agg["hm_mIoU"]["std"],
        }
        rows.append(row)

json_path = os.path.join(root, "ablation_summary.json")
csv_path = os.path.join(root, "ablation_summary.csv")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[SUMMARY] {param}")
for row in rows:
    print(
        f"  value={row['value']:>8s} K={row['k_shot']} | "
        f"mIoU={row['mIoU_mean']:.4f}±{row['mIoU_std']:.4f} | "
        f"Base={row['base_mIoU_mean']:.4f}±{row['base_mIoU_std']:.4f} | "
        f"Novel={row['novel_mIoU_mean']:.4f}±{row['novel_mIoU_std']:.4f} | "
        f"HM={row['hm_mIoU_mean']:.4f}±{row['hm_mIoU_std']:.4f}"
    )
print(f"  JSON: {json_path}")
print(f"  CSV : {csv_path}")
PY
}

# -----------------------------------------------------------------------------
# Execute all requested ablations sequentially.
# -----------------------------------------------------------------------------
echo "============================================================"
echo "ScanNet full hyperparameter ablation"
echo "Output root : ${ABLATION_ROOT}"
echo "Ablations  : ${ABLATIONS}"
echo "K list     : ${K_LIST}"
echo "GPUs/run   : ${GPU_LIST} (${NUM_GPUS})"
echo "Global seed: ${GLOBAL_SEED}"
echo "Baseline   : r=${BASE_R}, lambda_disc=${BASE_LAMBDA_DISC}, rho=${BASE_RHO}, M=${BASE_M}, lambda_v=${BASE_LAMBDA_V}"
echo "============================================================"

for param in "${ABLATION_NAMES[@]}"; do
  value_string="$(values_for_param "${param}")"
  read -r -a param_values <<< "${value_string}"
  for value in "${param_values[@]}"; do
    run_setting "${param}" "${value}"
  done
  summarize_parameter "${param}"
done

echo ""
echo "===================================================================="
echo "All requested ScanNet hyperparameter ablations completed."
echo "Root: ${ABLATION_ROOT}"
echo "Each parameter directory contains ablation_summary.json/.csv"
echo "===================================================================="
