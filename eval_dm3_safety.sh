#!/bin/bash

if [ $# -lt 4 ]; then
    echo "Usage: $0 <carla_port> <gpu_device> <checkpoint_path> <run_name> [additional_eval_parameters]"
    exit 1
fi

CARLA_PORT=$1
GPU_DEVICE=$2
CHECKPOINT_PATH=$3
RUN_NAME=$4
LOG_FILE="eval_log_${CARLA_PORT}_${RUN_NAME}.log"

ADDITIONAL_PARAMS="${@:5}"
CARLA_SERVER_COMMAND="$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10"
# EVAL_SCRIPT="dreamerv3/eval_safety.py"
COMMON_PARAMS="--env.world.carla_port $CARLA_PORT --dreamerv3.jax.policy_devices $GPU_DEVICE --dreamerv3.run.from_checkpoint $CHECKPOINT_PATH"
# ADDITIONAL_PARAMS="${@:4}"  # Capture all additional parameters passed to the script
# EVAL_COMMAND="python -u $EVAL_SCRIPT $COMMON_PARAMS $ADDITIONAL_PARAMS"

EVAL_MODULE="dreamerv3.eval_safety"
EVAL_MATCH="dreamerv3\.eval_safety"  #"python -m $EVAL_MODULE"
export PYTHONPATH="$PWD:$PYTHONPATH"   # ensure repo root on sys.path
EVAL_COMMAND="python -u -m $EVAL_MODULE $COMMON_PARAMS $ADDITIONAL_PARAMS"

# Clear log file before starting
> $LOG_FILE

# Function to log messages with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

# Function to start or restart CARLA
launch_carla() {
    # Check if CARLA is running
    if ! pgrep -f "CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10" > /dev/null; then
        log_with_timestamp "CARLA server is not running on port $CARLA_PORT. Starting or restarting..."
        # Kill any existing CARLA processes on the same port
        fuser -k ${CARLA_PORT}/tcp
        # Start CARLA
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE $CARLA_SERVER_COMMAND &
        # Wait for CARLA to fully start
        while ! nc -z localhost $CARLA_PORT; do
            log_with_timestamp "Waiting for CARLA server to start on port $CARLA_PORT..."
            sleep 1  # delay to prevent excessive resource usage
        done
        log_with_timestamp "CARLA server is up and running on port $CARLA_PORT."
    fi
}

# Function to start the eval script
start_eval() {
    launch_carla
    # Start the eval script
    $EVAL_COMMAND >> $LOG_FILE 2>&1 &
    EVAL_PID=$!
    # Log the information about the log file
    log_with_timestamp "Eval session started successfully. Logs are being written to: $LOG_FILE"
    # echo -e "\033[1;32mEval session started successfully. Logs are being written to: $LOG_FILE\033[0m"
}

# Function to clean up processes on exit
cleanup() {
    log_with_timestamp "Cleaning up and exiting..."
    # Kill CARLA process
    fuser -k ${CARLA_PORT}/tcp
    # Kill the specific eval process using its PID
    kill -TERM $EVAL_PID >/dev/null 2>&1
    wait $EVAL_PID >/dev/null 2>&1
    exit
}

# Trap EXIT signal to call the cleanup function
trap cleanup SIGINT

# Initial start
log_with_timestamp "Starting eval on port $CARLA_PORT..."
log_with_timestamp "Eval command: $EVAL_COMMAND"
start_eval


# Run once and exit (sweep-friendly)
wait $EVAL_PID
log_with_timestamp "Eval finished for run_name=$RUN_NAME"
exit 0