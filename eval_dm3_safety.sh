#!/bin/bash

set -uo pipefail   # (no -e: we want to handle failures ourselves)

if [ $# -lt 4 ]; then
  echo "Usage: $0 <carla_port> <gpu_device> <checkpoint_path> <run_name> [additional_eval_parameters]"
  exit 1
fi

CARLA_PORT="$1"
GPU_DEVICE="$2"
CHECKPOINT_PATH="$3"
RUN_NAME="$4"
shift 4
ADDITIONAL_PARAMS=( "$@" )

EVAL_MODULE="dreamerv3.eval_safety"
export PYTHONPATH="$PWD:$PYTHONPATH"

# --- knobs (override via env if needed) ---
MAX_RETRIES="${MAX_RETRIES:-3}"                 # how many times to attempt this run
CARLA_READY_TIMEOUT_S="${CARLA_READY_TIMEOUT_S:-180}"
CARLA_WARMUP_S="${CARLA_WARMUP_S:-5}"
KEEP_CARLA_RUNNING="${KEEP_CARLA_RUNNING:-1}"   # 1 = do not kill CARLA after run
RESTART_CARLA_ON_FAIL="${RESTART_CARLA_ON_FAIL:-1}"

# --- parse logdir and steps from passed flags (so wrapper can validate outputs) ---
LOGDIR=""
TARGET_STEPS=""
for ((i=0; i<${#ADDITIONAL_PARAMS[@]}; i++)); do
  if [[ "${ADDITIONAL_PARAMS[$i]}" == "--dreamerv3.logdir" ]] && (( i+1 < ${#ADDITIONAL_PARAMS[@]} )); then
    LOGDIR="${ADDITIONAL_PARAMS[$((i+1))]}"
  fi
  if [[ "${ADDITIONAL_PARAMS[$i]}" == "--dreamerv3.run.steps" ]] && (( i+1 < ${#ADDITIONAL_PARAMS[@]} )); then
    TARGET_STEPS="${ADDITIONAL_PARAMS[$((i+1))]}"
  fi
done

# Fallback: keep logs in CWD if logdir was not provided
if [[ -z "$LOGDIR" ]]; then
  LOGDIR="."
fi

mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/eval_log_${CARLA_PORT}_${RUN_NAME}.log"
: > "$LOG_FILE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE" >/dev/null; }

CARLA_SERVER_COMMAND=( "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen "-carla-port=${CARLA_PORT}" -benchmark -fps=10 )

EVAL_COMMAND=(
  python -u -m "$EVAL_MODULE"
  --env.world.carla_port "$CARLA_PORT"
  --dreamerv3.jax.policy_devices "$GPU_DEVICE"
  --dreamerv3.run.from_checkpoint "$CHECKPOINT_PATH"
  "${ADDITIONAL_PARAMS[@]}"
)

kill_carla() {
  # Kill whatever holds the port, then wait until the port is actually closed.
  fuser -k "${CARLA_PORT}/tcp" >/dev/null 2>&1 || true
  # Give UE a moment to die and release resources.
  for _ in {1..30}; do
    if ! nc -z 127.0.0.1 "$CARLA_PORT" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 0
}

start_carla() {
  log "Starting CARLA on port ${CARLA_PORT} ..."
  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "${CARLA_SERVER_COMMAND[@]}" >>"$LOG_FILE" 2>&1 &
  CARLA_PID=$!
  disown "$CARLA_PID" >/dev/null 2>&1 || true
}

wait_for_carla_ready() {
  # 1) wait for port open
  local t0
  t0="$(date +%s)"
  while ! nc -z 127.0.0.1 "$CARLA_PORT" >/dev/null 2>&1; do
    if (( $(date +%s) - t0 > CARLA_READY_TIMEOUT_S )); then
      log "ERROR: CARLA port did not open within ${CARLA_READY_TIMEOUT_S}s"
      return 1
    fi
    sleep 1
  done

  # 2) wait for CARLA "world ready" (port open is not enough)
  # This uses the same Python env you run eval with; if import fails, we fall back to warmup sleep.
  if python - <<PY >/dev/null 2>&1
import time
import carla
client = carla.Client("127.0.0.1", int("${CARLA_PORT}"))
client.set_timeout(2.0)
deadline = time.time() + float("${CARLA_READY_TIMEOUT_S}")
while time.time() < deadline:
    try:
        w = client.get_world()
        _ = w.get_map().name
        break
    except Exception:
        time.sleep(1.0)
else:
    raise SystemExit(2)
PY
  then
    log "CARLA world is ready."
  else
    log "WARN: CARLA Python readiness check failed; using warmup sleep only."
  fi

  sleep "$CARLA_WARMUP_S"
  return 0
}

ensure_carla() {
  # If port not open, start. If port open but not ready, restart.
  if ! nc -z 127.0.0.1 "$CARLA_PORT" >/dev/null 2>&1; then
    start_carla
  fi

  if ! wait_for_carla_ready; then
    log "WARN: CARLA not ready; restarting..."
    kill_carla
    start_carla
    wait_for_carla_ready
  fi
}

csv_nonempty() {
  local csv_path="$1"
  [[ -f "$csv_path" ]] || return 1
  # must have at least header + 1 row
  local lines
  lines="$(wc -l < "$csv_path" 2>/dev/null || echo 0)"
  [[ "$lines" -ge 2 ]]
}

run_success_outputs_present() {
  # Minimal robust condition: eval_episode_metrics.csv exists and has at least one episode row.
  # (If you have eval_done.json, you can strengthen this check.)
  local ep_csv="${LOGDIR}/eval_episode_metrics.csv"
  if csv_nonempty "$ep_csv"; then
    return 0
  fi
  return 1
}

# ---------------- main ----------------
log "Starting eval wrapper. RUN_NAME=${RUN_NAME}"
log "LOGDIR=${LOGDIR}"
log "EVAL_COMMAND: ${EVAL_COMMAND[*]}"

attempt=1
while (( attempt <= MAX_RETRIES )); do
  log "Attempt ${attempt}/${MAX_RETRIES}"

  ensure_carla

  # Clean obvious partial outputs before retrying (prevents appending mixed runs)
  rm -f "${LOGDIR}/eval_episode_metrics.csv" "${LOGDIR}/eval_done.json" >/dev/null 2>&1 || true

  log "Launching eval..."
  "${EVAL_COMMAND[@]}" >>"$LOG_FILE" 2>&1
  EVAL_RC=$?
  log "Eval process exited rc=${EVAL_RC}"

  if run_success_outputs_present; then
    log "OK: outputs present for ${RUN_NAME}"
    if [[ "$KEEP_CARLA_RUNNING" -ne 1 ]]; then
      log "Stopping CARLA (KEEP_CARLA_RUNNING=0)"
      kill_carla
    fi
    exit 0
  fi

  log "FAIL: outputs missing/empty for ${RUN_NAME}"
  if [[ "$RESTART_CARLA_ON_FAIL" -eq 1 ]]; then
    log "Restarting CARLA due to failure..."
    kill_carla
    start_carla
  fi

  attempt=$((attempt + 1))
done

log "ERROR: All retries failed for ${RUN_NAME}. Leaving logs in ${LOG_FILE}"
if [[ "$KEEP_CARLA_RUNNING" -ne 1 ]]; then
  kill_carla
fi
exit 2



# #!/bin/bash

# if [ $# -lt 4 ]; then
#     echo "Usage: $0 <carla_port> <gpu_device> <checkpoint_path> <run_name> [additional_eval_parameters]"
#     exit 1
# fi

# CARLA_PORT=$1
# GPU_DEVICE=$2
# CHECKPOINT_PATH=$3
# RUN_NAME=$4
# LOG_FILE="eval_log_${CARLA_PORT}_${RUN_NAME}.log"

# ADDITIONAL_PARAMS="${@:5}"
# CARLA_SERVER_COMMAND="$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10"
# # EVAL_SCRIPT="dreamerv3/eval_safety.py"
# COMMON_PARAMS="--env.world.carla_port $CARLA_PORT --dreamerv3.jax.policy_devices $GPU_DEVICE --dreamerv3.run.from_checkpoint $CHECKPOINT_PATH"
# # ADDITIONAL_PARAMS="${@:4}"  # Capture all additional parameters passed to the script
# # EVAL_COMMAND="python -u $EVAL_SCRIPT $COMMON_PARAMS $ADDITIONAL_PARAMS"

# EVAL_MODULE="dreamerv3.eval_safety"
# EVAL_MATCH="dreamerv3\.eval_safety"  #"python -m $EVAL_MODULE"
# export PYTHONPATH="$PWD:$PYTHONPATH"   # ensure repo root on sys.path
# EVAL_COMMAND="python -u -m $EVAL_MODULE $COMMON_PARAMS $ADDITIONAL_PARAMS"

# # Clear log file before starting
# > $LOG_FILE

# # Function to log messages with timestamp
# log_with_timestamp() {
#     echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
# }

# # Function to start or restart CARLA
# launch_carla() {
#     # Check if CARLA is running
#     if ! pgrep -f "CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10" > /dev/null; then
#         log_with_timestamp "CARLA server is not running on port $CARLA_PORT. Starting or restarting..."
#         # Kill any existing CARLA processes on the same port
#         fuser -k ${CARLA_PORT}/tcp
#         # Start CARLA
#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE $CARLA_SERVER_COMMAND &
#         # Wait for CARLA to fully start
#         while ! nc -z localhost $CARLA_PORT; do
#             log_with_timestamp "Waiting for CARLA server to start on port $CARLA_PORT..."
#             sleep 1  # delay to prevent excessive resource usage
#         done
#         log_with_timestamp "CARLA server is up and running on port $CARLA_PORT."
#     fi
# }

# # Function to start the eval script
# start_eval() {
#     launch_carla
#     # Start the eval script
#     $EVAL_COMMAND >> $LOG_FILE 2>&1 &
#     EVAL_PID=$!
#     # Log the information about the log file
#     log_with_timestamp "Eval session started successfully. Logs are being written to: $LOG_FILE"
#     # echo -e "\033[1;32mEval session started successfully. Logs are being written to: $LOG_FILE\033[0m"
# }

# # Function to clean up processes on exit
# cleanup() {
#     log_with_timestamp "Cleaning up and exiting..."
#     # Kill CARLA process
#     fuser -k ${CARLA_PORT}/tcp
#     # Kill the specific eval process using its PID
#     kill -TERM $EVAL_PID >/dev/null 2>&1
#     wait $EVAL_PID >/dev/null 2>&1
#     exit
# }

# # Trap EXIT signal to call the cleanup function
# trap cleanup SIGINT

# # Initial start
# log_with_timestamp "Starting eval on port $CARLA_PORT..."
# log_with_timestamp "Eval command: $EVAL_COMMAND"
# start_eval


# # Run once and exit (sweep-friendly)
# # wait $EVAL_PID
# # log_with_timestamp "Eval finished for run_name=$RUN_NAME"
# # exit 0

# wait $EVAL_PID
# EVAL_RC=$?
# log_with_timestamp "Eval finished for run_name=$RUN_NAME (rc=$EVAL_RC)"

# # optional: always restart CARLA between runs to reduce long-run UE4 instability
# fuser -k ${CARLA_PORT}/tcp >/dev/null 2>&1 || true

# exit $EVAL_RC

