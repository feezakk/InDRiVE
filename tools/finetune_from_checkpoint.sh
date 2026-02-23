#!/usr/bin/env bash
set -euo pipefail

# Finetune from a pretrain checkpoint for a single task and shield setting.
# Usage: ./tools/finetune_from_checkpoint.sh /path/to/checkpoint.ckpt carla_left_turn one_step

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <pretrain_ckpt> <task> <shield:noop|one_step|long> [out_dir]"
  exit 2
fi

PRETRAIN_CKPT="$1"
TASK="$2"
SHIELD="$3"
OUT_DIR="${4:-./experiments/finetune_${TASK}_${SHIELD}}"
mkdir -p "${OUT_DIR}"

# FAST mode short run
if [ "${FAST:-0}" = "1" ]; then
  FINETUNE_STEPS=${FINETUNE_STEPS:-50}
  EVAL_EPS=${EVAL_EPS:-1}
  echo "FAST mode: FINETUNE_STEPS=${FINETUNE_STEPS}, EVAL_EPS=${EVAL_EPS}"
else
  FINETUNE_STEPS=${FINETUNE_STEPS:-10000}
  EVAL_EPS=${EVAL_EPS:-3}
fi

# Optional debug mode for JAX/CuDNN issues
if [ "${JAX_DEBUG:-0}" = "1" ]; then
  echo "JAX_DEBUG=1: disabling autotune and GPU preallocation/mixed-precision"
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export TF_ENABLE_AUTO_MIXED_PRECISION=0
fi

case "${SHIELD}" in
  noop)
    SHIELD_FLAGS=(--dreamerv3.safe_train.enable False)
    ;;
  one_step)
    SHIELD_FLAGS=(--dreamerv3.safe_train.enable True --dreamerv3.safe_train.mode one_step)
    ;;
  long)
    SHIELD_FLAGS=(--dreamerv3.safe_train.enable True --dreamerv3.safe_train.mode long --dreamerv3.safe_train.horizon 15)
    ;;
  *)
    echo "Unknown shield: ${SHIELD}. Use noop|one_step|long"
    exit 2
    ;;
esac

echo "=== Finetune from ${PRETRAIN_CKPT} -> task=${TASK} shield=${SHIELD} out=${OUT_DIR} ==="
python -u -m dreamerv3.train \
  --dreamerv3.run.steps ${FINETUNE_STEPS} \
  --dreamerv3.logdir ${OUT_DIR} \
  --dreamerv3.run.eval_eps ${EVAL_EPS} \
  --task ${TASK} \
  --from_checkpoint ${PRETRAIN_CKPT} \
  --env.world.carla_port 2000 \
  "${SHIELD_FLAGS[@]}"

echo "Finetune finished. Results in ${OUT_DIR}"
