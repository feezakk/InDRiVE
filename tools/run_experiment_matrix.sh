#!/usr/bin/env bash
set -euo pipefail

# Run the full experiment matrix described by the user:
# - Pretrain (intrinsic-only)
# - Calibrate collision head on held-out val set
# - Fine-tune variants (no-shield/shield x head frozen/fine-tuned)
# - Zero-shot and few-shot evaluations
# Usage: PRETRAIN_STEPS=200 FINETUNE_STEPS=50 ./tools/run_experiment_matrix.sh

CKPT_ROOT="./experiments"
mkdir -p "$CKPT_ROOT"

# Basic defaults (override via env vars)
K=${K:-5}                     # collision horizon
VAL_EPISODES=${VAL_EPISODES:-100}
TEST_EPISODES=${TEST_EPISODES:-300}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TASKS=${TASKS:-"carla_lane_following carla_left_turn"}
SHIELD=${SHIELD:-one_step}    # primary shield for matrix (one_step)
FAST=${FAST:-0}

# Steps defaults forwarded to per-step scripts
PRETRAIN_STEPS=${PRETRAIN_STEPS:-${DEFAULT_PRETRAIN_STEPS:-1000000}}
FINETUNE_STEPS=${FINETUNE_STEPS:-${DEFAULT_FINETUNE_STEPS:-10000}}

echo "Experiment matrix: K=${K} VAL_EPISODES=${VAL_EPISODES} TEST_EPISODES=${TEST_EPISODES} SEEDS=${SEEDS} TASKS=${TASKS}"

# 1) Pretrain intrinsic-only (single run across seeds)
echo "--- Pretraining (intrinsic-only) ---"
for s in ${SEEDS}; do
  PRE_DIR="${CKPT_ROOT}/pretrain_ld_s${s}"
  mkdir -p "${PRE_DIR}"
  echo "Pretraining seed=${s} -> ${PRE_DIR}"
  PRETRAIN_CMD=(FAST=${FAST} PRETRAIN_TASK=carla_lane_following PRETRAIN_STEPS=${PRETRAIN_STEPS} ./tools/pretrain_latent_disagreement.sh)
  ( set -x; SEED=${s} "${PRETRAIN_CMD[@]}" )
done

echo "Pretrain finished. Proceeding to calibration and finetune matrix."

# 2) Calibration: run val episodes for collision head and compute temperature scaling
echo "--- Calibration (collect validation episodes) ---"
CAL_DIR="${CKPT_ROOT}/calibration"
mkdir -p "${CAL_DIR}"
for s in ${SEEDS}; do
  CKPT="${CKPT_ROOT}/pretrain_ld_s${s}/checkpoint.ckpt"
  echo "Collecting val episodes for seed ${s} using ${CKPT}"
  # run evaluation to collect per-episode CSVs (use tools/eval_from_checkpoint.sh but with FAST/VAL_EPISODES)
  FAST=${FAST} EVAL_EPS=${VAL_EPISODES} ./tools/eval_from_checkpoint.sh "${CKPT}" carla_lane_following noop "${CAL_DIR}/seed_${s}"
done

echo "Run calibration: temperature-scaling"
python tools/calibrate_collision.py --in_dir "${CAL_DIR}" --out_dir "${CAL_DIR}/calibration_results.json" --k ${K}

# 3) Fine-tune matrix: for each task, each shield, and each head-freeze option
echo "--- Fine-tune matrix ---"
for s in ${SEEDS}; do
  for task in ${TASKS}; do
    for shield in noop ${SHIELD}; do
      for head_mode in frozen finetuned; do
        FIN_DIR="${CKPT_ROOT}/finetune_${task}_${shield}_head-${head_mode}_s${s}"
        mkdir -p "${FIN_DIR}"
        echo "Finetune seed=${s} task=${task} shield=${shield} head=${head_mode} -> ${FIN_DIR}"
        # Prepare flags for head freeze
        if [ "${head_mode}" = "frozen" ]; then
          HEAD_FLAGS=(--dreamerv3.train.freeze_collision_head True)
        else
          HEAD_FLAGS=()
        fi
        # Prepare worldmodel freeze for zero-shot variant later
        ( set -x; SEED=${s} JAX_DEBUG=${JAX_DEBUG:-0} ./tools/finetune_from_checkpoint.sh "${CKPT_ROOT}/pretrain_ld_s${s}/checkpoint.ckpt" ${task} ${shield} )
      done
    done
  done
done

echo "Fine-tune stage submitted. After finetuning, run zero-shot and few-shot evals using the produced checkpoints and then generate reports with tools/generate_experiment_report.py"
