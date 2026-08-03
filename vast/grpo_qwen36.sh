#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

source /venv/main/bin/activate
cd /workspace/grpo_poc
exec python -u grpo_poc_qwen36.py --phase grpo
