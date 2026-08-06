#!/usr/bin/env bash
set -euo pipefail

# LORA_ADAPTER_PATH must be the lora_adapter directory made by VERL's model merger.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ID=${MODEL_ID:-${MODEL_DIR:-/dev/shm/gpt-oss-20b-bf16}}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:?Set LORA_ADAPTER_PATH to the SFT lora_adapter directory}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/checkpoints/grpo"}

# The dequantized 20B FSDP actor occupies roughly 45 GiB on this 93 GiB H100. Leave
# headroom for it while the colocated vLLM rollout server owns its KV cache.
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=dr_grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$ROOT/data/grpo_train.parquet" \
  data.val_files="$ROOT/data/grpo_val.parquet" \
  data.train_batch_size=4 \
  data.gen_batch_size=1 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  +data.apply_chat_template_kwargs.reasoning_effort=medium \
  actor_rollout_ref.model.path="$MODEL_ID" \
  actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH" \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=3e-5 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_model_len=768 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
  actor_rollout_ref.rollout.layered_summon=true \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024 \
  reward.custom_reward_function.path="$ROOT/reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  reward.num_workers=1 \
  trainer.logger='["console"]' \
  trainer.project_name=revenue_json_lora \
  trainer.experiment_name=grpo \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=25 \
  trainer.total_training_steps=100 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.resume_mode=disable \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1
