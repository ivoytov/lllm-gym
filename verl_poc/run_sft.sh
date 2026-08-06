#!/usr/bin/env bash
set -euo pipefail

# Run from this directory after installing VERL. One H100 GPU, LoRA only.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ID=${MODEL_ID:-${MODEL_DIR:-/dev/shm/gpt-oss-20b-bf16}}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/checkpoints/sft"}

torchrun --standalone --nproc_per_node=1 -m verl.trainer.sft_trainer \
  data.train_files="$ROOT/data/sft.parquet" \
  data.messages_key=messages \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=true \
  data.train_batch_size=8 \
  data.micro_batch_size_per_gpu=1 \
  data.max_token_len_per_gpu=2048 \
  data.use_dynamic_bsz=true \
  data.ignore_input_ids_mismatch=true \
  +data.apply_chat_template_kwargs.reasoning_effort=medium \
  model.path="$MODEL_ID" \
  model.lora_rank=32 \
  model.lora_alpha=64 \
  model.target_modules=all-linear \
  +model.override_config.attn_implementation=eager \
  model.enable_gradient_checkpointing=true \
  model.use_remove_padding=true \
  engine.model_dtype=bfloat16 \
  optim.lr=1e-4 \
  trainer.total_epochs=2 \
  trainer.save_freq=48 \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.project_name=revenue_json_lora \
  trainer.experiment_name=sft \
  trainer.logger='["console"]' \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node=1
