#!/usr/bin/env bash

set -x

umask 007


export LINGBOT_DEVICE=musa

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-your key}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-your url}"
export WANDB_TEAM_NAME="${WANDB_TEAM_NAME:-your team name}"
export WANDB_PROJECT="${WANDB_PROJECT:-your project}"

num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LINGBOT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LINGBOT_ROOT=${LINGBOT_ROOT:-${DEFAULT_LINGBOT_ROOT}}
export LINGBOT_ROOT
export PYTHONPATH="${LINGBOT_ROOT}/wan_va:${PYTHONPATH:-}"
cd "${LINGBOT_ROOT}/wan_va"

TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python3 -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m train --config-name ${config_name} $overrides
