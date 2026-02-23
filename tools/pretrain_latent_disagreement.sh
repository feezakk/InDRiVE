#!/usr/bin/env bash
set -euo pipefail

# Pretrain latent-disagreement only (intrinsic reward) helper script.
# Usage: FAST=1 PRETRAIN_STEPS=200 ./tools/pretrain_latent_disagreement.sh

CKPT_ROOT="./experiments"
mkdir -p "$CKPT_ROOT"

# Optional: set JAX_DEBUG=1 to reduce cudnn autotune overhead and GPU preallocation
# This helps when you see long conv autotune logs or allocator OOM messages.
if [ "${JAX_DEBUG:-0}" = "1" ]; then
  echo "JAX_DEBUG=1: disabling autotune and GPU preallocation/mixed-precision"
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export TF_ENABLE_AUTO_MIXED_PRECISION=0
fi

CARLA_HOME=${CARLA_HOME:-/home/bimi2/bimians/feeza/CARLA_0.9.15}
CARLA_LOG="${CKPT_ROOT}/carla.log"
CARLA_HOST=${CARLA_HOST:-127.0.0.1}
CARLA_PORT=${CARLA_PORT:-2000}
TIMEOUT=${CARLA_WAIT_TIMEOUT:-300}
ELAPSED=0
SLEEP=2

echo "Waiting for CARLA at ${CARLA_HOST}:${CARLA_PORT} (timeout ${TIMEOUT}s)..."
start_carla_if_needed() {
  if [ -z "${CARLA_HOME:-}" ] || [ ! -d "${CARLA_HOME}" ]; then
    return 0
  fi
  candidates=("$CARLA_HOME/CarlaUE4.sh" "$CARLA_HOME/CarlaUE4/CarlaUE4.sh" "$CARLA_HOME/CarlaUE4/Binaries/Linux/CarlaUE4" "$CARLA_HOME/CarlaUE4.sh")
  for c in "${candidates[@]}"; do
    if [ -x "$c" ]; then
      echo "Found CARLA executable: $c"
      echo "Starting CARLA (logging to ${CARLA_LOG})..."
      mkdir -p "$(dirname "$CARLA_LOG")"
      if [ -z "${DISPLAY-}" ]; then
        if command -v xvfb-run >/dev/null 2>&1; then
          nohup xvfb-run -s '"-screen 0 1280x720x24"' "$c" -world-port=${CARLA_PORT} &> "${CARLA_LOG}" &
        else
          nohup "$c" -world-port=${CARLA_PORT} &> "${CARLA_LOG}" &
        fi
      else
        nohup "$c" -world-port=${CARLA_PORT} &> "${CARLA_LOG}" &
      fi
      sleep 6
      return 0
    fi
  done
  echo "CARLA_HOME is set to ${CARLA_HOME} but no executable was found in expected locations. Skipping auto-start."
}

start_carla_if_needed
while true; do
  if command -v nc >/dev/null 2>&1; then
    if nc -z ${CARLA_HOST} ${CARLA_PORT} >/dev/null 2>&1; then
      break
    fi
  else
    python - <<PY >/dev/null 2>&1 || true
import socket,sys
s=socket.socket()
s.settimeout(1)
try:
    s.connect(('${CARLA_HOST}', ${CARLA_PORT}))
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
    if [ $? -eq 0 ]; then
      break
    fi
  fi
  sleep ${SLEEP}
  ELAPSED=$((ELAPSED + SLEEP))
  echo "  still waiting... ${ELAPSED}s elapsed"
  if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
    echo "Timed out waiting for CARLA after ${TIMEOUT}s"
    exit 1
  fi
done
echo "CARLA is up on ${CARLA_HOST}:${CARLA_PORT}"

# FAST mode defaults
if [ "${FAST:-0}" = "1" ]; then
  PRETRAIN_STEPS=${PRETRAIN_STEPS:-200}
  echo "FAST mode: PRETRAIN_STEPS=${PRETRAIN_STEPS}"
else
  PRETRAIN_STEPS=${PRETRAIN_STEPS:-1000000}
fi

PRETRAIN_TASK=${PRETRAIN_TASK:-carla_lane_following}
PRETRAIN_LOGDIR="$CKPT_ROOT/pretrain_ld"

echo "=== Pretraining latent-disagreement only (no external reward) ==="
# Allow an explicit override for pretrain shield; default is "noop" (disabled)
PRETRAIN_SHIELD=${PRETRAIN_SHIELD:-noop}
case "${PRETRAIN_SHIELD}" in
  noop)
    PRETRAIN_SHIELD_FLAGS=(--dreamerv3.safe_train.enable False)
    ;;
  one_step)
    PRETRAIN_SHIELD_FLAGS=(--dreamerv3.safe_train.enable True --dreamerv3.safe_train.mode one_step)
    ;;
  long)
    PRETRAIN_SHIELD_FLAGS=(--dreamerv3.safe_train.enable True --dreamerv3.safe_train.mode long --dreamerv3.safe_train.horizon 15)
    ;;
  *)
    echo "Unknown PRETRAIN_SHIELD=${PRETRAIN_SHIELD}; using noop"
    PRETRAIN_SHIELD_FLAGS=(--dreamerv3.safe_train.enable False)
    ;;
esac

python -u -m dreamerv3.train \
  --task ${PRETRAIN_TASK} \
  --dreamerv3.run.steps ${PRETRAIN_STEPS} \
  --dreamerv3.logdir ${PRETRAIN_LOGDIR} \
  --dreamerv3.expl_rewards.disag 1.0 \
  --dreamerv3.expl_rewards.extr 0.0 \
  --dreamerv3.run.eval_eps 0 \
  --env.world.carla_port ${CARLA_PORT} \
  "${PRETRAIN_SHIELD_FLAGS[@]}"

echo "Pretraining finished. Checkpoints and logs in ${PRETRAIN_LOGDIR}"
