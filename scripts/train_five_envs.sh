#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <base_logdir> <num_procs> [extra_train_args...]"
  echo "Example: $0 ./experiments 12 --use_wandb --wandb_project HumanoidRL --wandb_video_freq 1"
  exit 1
fi

BASE_LOGDIR="$1"
NUM_PROCS="$2"
shift 2
EXTRA_ARGS=("$@")

ENVS=(
  "jvrc_run"
  "jvrc_run_arm"
  "jvrc_walk"
  "jvrc_step"
  "jvrc_arm"
)

mkdir -p "${BASE_LOGDIR}"

for ENV_NAME in "${ENVS[@]}"; do
  LOGDIR="${BASE_LOGDIR}/${ENV_NAME}"
  echo "=============================="
  echo "Training env: ${ENV_NAME}"
  echo "Logdir: ${LOGDIR}"
  echo "=============================="
  python run_experiment.py train \
    --env "${ENV_NAME}" \
    --logdir "${LOGDIR}" \
    --num_procs "${NUM_PROCS}" \
    "${EXTRA_ARGS[@]}"
done

echo "All 5 environments finished."
