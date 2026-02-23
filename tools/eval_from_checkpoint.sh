#!/usr/bin/env bash
set -euo pipefail

# Evaluate a checkpoint with safe_eval_mode and collect per-episode CSVs.
# Usage: ./tools/eval_from_checkpoint.sh /path/to/checkpoint.ckpt carla_left_turn one_step

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <checkpoint> <task> <safe_eval_mode:noop|one_step|long> [out_dir]"
  exit 2
fi

CKPT="$1"
TASK="$2"
MODE="$3"
OUT_DIR="${4:-./experiments/eval_${TASK}_${MODE}}"
mkdir -p "${OUT_DIR}"

EVAL_EPS=${EVAL_EPS:-3}
if [ "${FAST:-0}" = "1" ]; then
  EVAL_EPS=${EVAL_EPS:-1}
  echo "FAST mode: EVAL_EPS=${EVAL_EPS}"
fi

# Optional debug mode for JAX/CuDNN issues
if [ "${JAX_DEBUG:-0}" = "1" ]; then
  echo "JAX_DEBUG=1: disabling autotune and GPU preallocation/mixed-precision"
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export TF_ENABLE_AUTO_MIXED_PRECISION=0
fi

echo "=== Evaluate ${CKPT} task=${TASK} mode=${MODE} -> ${OUT_DIR} (eps=${EVAL_EPS}) ==="
python -m dreamerv3.eval_safety \
  --from_checkpoint ${CKPT} \
  --logdir ${OUT_DIR} \
  --env.world.carla_port 2000 \
  --task ${TASK} \
  --safe_eval_mode ${MODE} \
  --safe_eval_horizon 15 \
  --run.eval_eps ${EVAL_EPS}

# Optionally collect per-episode CSVs (one per episode)
for i in $(seq 0 $((EVAL_EPS-1))); do
  OUT_CSV="${OUT_DIR}/eval_ep_${i}_mode_${MODE}.csv"
  echo "Extracting episode ${i} -> ${OUT_CSV}"
  python tools/log_eval_episode.py --checkpoint ${CKPT} --port 2000 --task ${TASK} --out ${OUT_CSV} || true
done

echo "Evaluation finished. Results in ${OUT_DIR}"
