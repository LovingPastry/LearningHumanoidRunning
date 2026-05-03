#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <base_logdir> <num_procs> [--resume] [--continued_root <dir>] [extra_train_args...]"
  echo "Example: $0 ./experiments 12 --use_wandb --wandb_project HumanoidRL --wandb_video_freq 1"
  echo "Example (resume in-place): $0 ./experiments 12 --resume --use_wandb"
  echo "Example (resume from another root): $0 ./new_runs 12 --continued_root ./old_runs --use_wandb"
  exit 1
fi

BASE_LOGDIR="$1"
NUM_PROCS="$2"
shift 2

RESUME_MODE=0
CONTINUED_ROOT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_MODE=1
      shift
      ;;
    --continued_root)
      if [[ $# -lt 2 ]]; then
        echo "Error: --continued_root requires a directory path."
        exit 1
      fi
      CONTINUED_ROOT="$2"
      shift 2
      ;;
    --continued)
      echo "Error: do not pass --continued directly to train_five_envs.sh."
      echo "Use --resume or --continued_root <dir> so each environment loads its own checkpoint."
      exit 1
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${RESUME_MODE} -eq 1 && -n "${CONTINUED_ROOT}" ]]; then
  echo "Error: --resume and --continued_root cannot be used together."
  exit 1
fi

if [[ ${RESUME_MODE} -eq 1 ]]; then
  CONTINUED_ROOT="${BASE_LOGDIR}"
fi

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
  CONTINUED_ARGS=()
  if [[ -n "${CONTINUED_ROOT}" ]]; then
    RESUME_PATH="${CONTINUED_ROOT}/${ENV_NAME}"
    CONTINUED_ARGS=(--continued "${RESUME_PATH}")
  fi
  echo "=============================="
  echo "Training env: ${ENV_NAME}"
  echo "Logdir: ${LOGDIR}"
  if [[ ${#CONTINUED_ARGS[@]} -gt 0 ]]; then
    echo "Resume from: ${RESUME_PATH}"
  fi
  echo "=============================="
  python run_experiment.py train \
    --env "${ENV_NAME}" \
    --logdir "${LOGDIR}" \
    --num_procs "${NUM_PROCS}" \
    "${CONTINUED_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
done

echo "All 5 environments finished."
