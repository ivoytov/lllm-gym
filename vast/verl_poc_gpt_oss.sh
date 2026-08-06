#!/usr/bin/env bash
set -euo pipefail

# Supervisor entrypoint for the VERL GPT-OSS PoC on Vast.ai.
#
# This script is deliberately self-bootstrapping: a fresh container only needs
# git, Python, and the NVIDIA runtime. It clones the lllm-gym checkout, clones
# VERL, creates a tmpfs virtualenv, installs dependencies, materializes the
# BF16 GPT-OSS model in tmpfs, and then runs the SFT -> merge -> GRPO pipeline.
# Override any *_DIR, *_URL, *_REF, or *_REINSTALL variable when reproducing it.

set -eo pipefail
set +u
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -u

WORKSPACE=${WORKSPACE:-/workspace}
LLMS_DIR=${LLMS_DIR:-"$WORKSPACE/llms"}
LLMS_REPO_URL=${LLMS_REPO_URL:-https://github.com/ivoytov/lllm-gym.git}
LLMS_REF=${LLMS_REF:-main}
VERL_DIR=${VERL_DIR:-"$WORKSPACE/verl"}
VERL_REPO_URL=${VERL_REPO_URL:-https://github.com/verl-project/verl.git}
VERL_REF=${VERL_REF:-main}
PYENV_DIR=${PYENV_DIR:-/dev/shm/verl-venv}
HF_HOME=${HF_HOME:-/dev/shm/huggingface}
MODEL_DIR=${MODEL_DIR:-/dev/shm/gpt-oss-20b-bf16}

mkdir -p "$WORKSPACE" "$HF_HOME"
export HF_HOME MODEL_DIR PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HOME"}
# GPT-OSS uses attention sinks; this avoids an incompatible FlashInfer sampler.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

clone_once() {
  local destination=$1 url=$2 ref=$3
  if [[ ! -d "$destination/.git" ]]; then
    if [[ -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      # A local deploy may rsync a working tree without its .git directory.
      # Preserve that tree; a clean container takes the clone branch instead.
      echo "Using existing working tree at $destination"
    else
      git clone --depth 1 --branch "$ref" "$url" "$destination"
    fi
  elif [[ "${UPDATE_GIT_CHECKOUTS:-0}" == "1" ]]; then
    git -C "$destination" fetch --depth 1 origin "$ref"
    git -C "$destination" checkout --force FETCH_HEAD
  fi
}

clone_once "$LLMS_DIR" "$LLMS_REPO_URL" "$LLMS_REF"
clone_once "$VERL_DIR" "$VERL_REPO_URL" "$VERL_REF"

if [[ ! -x "$PYENV_DIR/bin/python" ]]; then
  rm -rf "$PYENV_DIR"
  /venv/main/bin/python -m venv --system-site-packages "$PYENV_DIR"
fi

if [[ "${REINSTALL_DEPS:-0}" == "1" || ! -f "$PYENV_DIR/.verl_gpt_oss_installed" ]]; then
  uv pip install --python "$PYENV_DIR/bin/python" --upgrade pip
  uv pip install --python "$PYENV_DIR/bin/python" -e "$VERL_DIR" "vllm>=0.18.0"
  touch "$PYENV_DIR/.verl_gpt_oss_installed"
fi

source "$PYENV_DIR/bin/activate"
cd "$LLMS_DIR"
exec bash "$LLMS_DIR/verl_poc/run_all.sh"
