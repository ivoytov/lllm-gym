#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 "$ROOT/patch_verl_qwen36.py"

if [[ ! -d "$ROOT/exports/sft/lora_adapter" ]]; then
  SFT_STEP=$(find "$ROOT/checkpoints/sft" -type d -name 'global_step_*' | sort -V | tail -n 1)
  test -n "$SFT_STEP"
  python3 -m verl.model_merger merge --backend fsdp \
    --local_dir "$SFT_STEP" \
    --target_dir "$ROOT/exports/sft"
fi

export LORA_ADAPTER_PATH="$ROOT/exports/sft/lora_adapter"
bash "$ROOT/run_grpo.sh"
