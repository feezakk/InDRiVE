#!/usr/bin/env bash
set -euo pipefail

# Orchestrate latent-disagreement pretrain + finetune experiments.
# Adjust paths, python/env activation as needed.

CKPT_ROOT="./experiments"
mkdir -p "$CKPT_ROOT"

# Optional: path to a local CARLA installation. If not set, the script will
# only wait for an externally-started CARLA. Default is the path you provided
# (adjust if your CARLA is elsewhere) — you can override by setting CARLA_HOME
# in the environment.
CARLA_HOME=${CARLA_HOME:-/home/bimi2/bimians/feeza/CARLA_0.9.15}
CARLA_LOG="${CKPT_ROOT}/carla.log"

# Wait for CARLA to be available before starting experiments.
# This tries `nc -z` if available, otherwise falls back to a small Python TCP connect check.
CARLA_HOST=${CARLA_HOST:-127.0.0.1}
CARLA_PORT=${CARLA_PORT:-2000}
TIMEOUT=${CARLA_WAIT_TIMEOUT:-300}
ELAPSED=0
SLEEP=2
echo "Waiting for CARLA at ${CARLA_HOST}:${CARLA_PORT} (timeout ${TIMEOUT}s)..."
# If CARLA is not reachable but CARLA_HOME exists, attempt to start it.
start_carla_if_needed() {
  if [ -z "${CARLA_HOME:-}" ] || [ ! -d "${CARLA_HOME}" ]; then
    return 0
  fi
  # possible executable locations
  candidates=("$CARLA_HOME/CarlaUE4.sh" "$CARLA_HOME/CarlaUE4/CarlaUE4.sh" "$CARLA_HOME/CarlaUE4/Binaries/Linux/CarlaUE4" "$CARLA_HOME/CarlaUE4.sh" )
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
      # give CARLA a few seconds to start
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
    # Fallback: use Python to test TCP connection
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

# Experiment matrix
# Default experiment sizes (can be overridden via env vars or FAST=1 for smoke tests)
DEFAULT_PRETRAIN_STEPS=1000000
DEFAULT_FINETUNE_STEPS=10000
TASKS=(carla_lane_following carla_left_turn carla_right_turn)
SHIELDS=(noop one_step long)

# FAST mode: set FAST=1 in the environment to shorten runs for pipeline verification.
# You can also manually set PRETRAIN_STEPS and FINETUNE_STEPS env vars to control sizes.
if [ "${FAST:-0}" = "1" ]; then
  PRETRAIN_STEPS=${PRETRAIN_STEPS:-200}
  FINETUNE_STEPS=${FINETUNE_STEPS:-50}
  EVAL_EPS=${EVAL_EPS:-1}
  echo "FAST mode enabled: PRETRAIN_STEPS=${PRETRAIN_STEPS}, FINETUNE_STEPS=${FINETUNE_STEPS}, EVAL_EPS=${EVAL_EPS}"
else
  PRETRAIN_STEPS=${PRETRAIN_STEPS:-$DEFAULT_PRETRAIN_STEPS}
  FINETUNE_STEPS=${FINETUNE_STEPS:-$DEFAULT_FINETUNE_STEPS}
  EVAL_EPS=${EVAL_EPS:-3}
fi

# Which task to use for pretraining (must provide the observables the model expects,
# e.g., 'offlane'). Override with PRETRAIN_TASK env var if needed.
PRETRAIN_TASK=${PRETRAIN_TASK:-carla_lane_following}

# Path to checkpoint after pretraining
PRETRAIN_LOGDIR="$CKPT_ROOT/pretrain_ld"

echo "=== Pretraining latent-disagreement only (no external reward) ==="
python -u -m dreamerv3.train \
  --task ${PRETRAIN_TASK} \
  --dreamerv3.run.steps ${PRETRAIN_STEPS} \
  --dreamerv3.logdir ${PRETRAIN_LOGDIR} \
  --dreamerv3.expl_rewards.disag 1.0 \
  --dreamerv3.expl_rewards.extr 0.0 \
  --dreamerv3.run.eval_eps 0 \
  --env.world.carla_port 2000

PRETRAIN_CKPT="${PRETRAIN_LOGDIR}/checkpoint.ckpt"

for shield in "${SHIELDS[@]}"; do
  for task in "${TASKS[@]}"; do
    FINETUNE_DIR="$CKPT_ROOT/finetune_${task}_${shield}"
    mkdir -p "$FINETUNE_DIR"
    echo "=== Finetune: task=$task shield=$shield ==="

    # Determine shield flags
    case "$shield" in
      noop)
        SHIELD_FLAGS=(--dreamerv3.safe_train.enable False)
        ;;
      one_step)
        SHIELD_FLAGS=(--dreamerv3.safe_train.enable True --dreamerv3.safe_train.mode one_step)
        ;;
      long)
        SHIELD_FLAGS=(--dreamerv3.safe_train.enable True --dreamerv3.safe_train.mode long --dreamerv3.safe_train.horizon 15)
        ;;
    esac

    python -u -m dreamerv3.train \
      --dreamerv3.run.steps ${FINETUNE_STEPS} \
      --dreamerv3.logdir ${FINETUNE_DIR} \
      --dreamerv3.run.eval_eps ${EVAL_EPS} \
      --task ${task} \
      --from_checkpoint ${PRETRAIN_CKPT} \
      --env.world.carla_port 2000 \
      "${SHIELD_FLAGS[@]}"

    echo "=== Evaluate finetuned model: task=$task shield=$shield ==="
    EVAL_DIR="$FINETUNE_DIR/eval"
    mkdir -p "$EVAL_DIR"
    python -m dreamerv3.embodied.run.eval_safety \
      --from_checkpoint ${FINETUNE_DIR}/checkpoint.ckpt \
      --logdir ${EVAL_DIR} \
      --env.world.carla_port 2000 \
      --task ${task} \
      --safe_eval_mode ${shield} \
      --safe_eval_horizon 15 \
      --run.eval_eps ${EVAL_EPS}

    # Collect per-episode CSVs with the lightweight helper (one file per episode)
    for i in 0 1 2; do
      OUT_CSV="${EVAL_DIR}/eval_ep_${i}_shield_${shield}.csv"
      python tools/log_eval_episode.py --checkpoint ${FINETUNE_DIR}/checkpoint.ckpt --port 2000 --task ${task} --out ${OUT_CSV}
    done
  done
done

echo "All experiments finished. Results in $CKPT_ROOT"
