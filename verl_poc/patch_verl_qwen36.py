#!/usr/bin/env python3
"""Compatibility patch for Qwen3.6 configuration names in current VERL."""

from pathlib import Path
import sys


verl_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/verl")
target = verl_root / "verl/models/transformers/qwen3_5.py"
source = target.read_text()

# Qwen3.5 exposed ``layer_type`` and top-level ``num_hidden_layers``. Qwen3.6
# puts these on the nested text config as ``block_type`` and `text_config`.
patches = {
    'getattr(self, "layer_type", "full_attention")': 'getattr(self, "layer_type", self.block_type)',
}

for old, new in patches.items():
    if old in source:
        source = source.replace(old, new)
    elif new not in source:
        raise RuntimeError(f"Expected Qwen layer-type expression was not found in {target}")

target.write_text(source)

vllm_target = verl_root / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
vllm_source = vllm_target.read_text()
old = "self.model_config.hf_config.num_hidden_layers"
new = "getattr(self.model_config.hf_config, 'num_hidden_layers', self.model_config.hf_config.text_config.num_hidden_layers)"
if old in vllm_source:
    vllm_target.write_text(vllm_source.replace(old, new))
elif new not in vllm_source:
    raise RuntimeError(f"Expected Qwen hidden-layer expression was not found in {vllm_target}")

# Qwen3.6 mixes 4304- and 12288-wide projections. Their common divisor is
# only 16, while the FP8 Triton kernel requires an inner block of at least
# 32. Use per-tensor FP8 instead of VERL's incompatible block quantization.
vllm_source = vllm_target.read_text()
for old in ('"weight_block_size": [128, 128],\n', '"weight_block_size": [16, 16],\n'):
    vllm_source = vllm_source.replace(old, '')
if '"weight_block_size"' in vllm_source:
    raise RuntimeError(f"Unexpected FP8 block-size expression in {vllm_target}")
vllm_target.write_text(vllm_source)

print(f"Patched Qwen3.6 compatibility: {target} and {vllm_target}")
