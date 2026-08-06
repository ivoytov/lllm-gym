#!/usr/bin/env bash
set -euo pipefail

# Run this from the local lllm-gym checkout. It copies the current working tree
# (including uncommitted changes), installs the supervisor wrapper/config, and
# starts the managed pipeline. A clean container can instead run
# /opt/supervisor-scripts/verl_poc_gpt_oss.sh and clone the repository itself.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SSH_HOST=${SSH_HOST:-root@20.33.104.13}
SSH_PORT=${SSH_PORT:-40486}
REMOTE_WORKSPACE=${REMOTE_WORKSPACE:-/workspace}
REMOTE_LLMS_DIR=${REMOTE_LLMS_DIR:-"$REMOTE_WORKSPACE/llms"}

SSH=(ssh -p "$SSH_PORT")
SCP=(scp -P "$SSH_PORT")

"${SSH[@]}" "$SSH_HOST" "mkdir -p '$REMOTE_LLMS_DIR'"
rsync -az \
  --exclude '.git/' \
  --exclude '**/__pycache__/' \
  --exclude 'checkpoints/' \
  --exclude 'exports/' \
  --exclude 'models/' \
  --exclude 'data/*.parquet' \
  -e "ssh -p $SSH_PORT" \
  "$ROOT/" "$SSH_HOST:$REMOTE_LLMS_DIR/"

"${SCP[@]}" "$ROOT/vast/verl_poc_gpt_oss.sh" "$SSH_HOST:/tmp/verl_poc_gpt_oss.sh"
"${SCP[@]}" "$ROOT/vast/verl_poc_gpt_oss.conf" "$SSH_HOST:/tmp/verl_poc_gpt_oss.conf"
"${SSH[@]}" "$SSH_HOST" 'install -m 755 /tmp/verl_poc_gpt_oss.sh /opt/supervisor-scripts/verl_poc_gpt_oss.sh && install -m 644 /tmp/verl_poc_gpt_oss.conf /etc/supervisor/conf.d/verl_poc_gpt_oss.conf && supervisorctl reread && supervisorctl update && supervisorctl restart verl_poc_gpt_oss'
