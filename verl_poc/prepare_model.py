#!/usr/bin/env python3
"""Materialize a trainable BF16 copy of openai/gpt-oss-20b.

The Hub checkpoint is MXFP4-quantized.  VERL's FSDP/LoRA training path needs
the MoE weights dequantized first, while vLLM can load the resulting safetensors
directly for rollout.  MODEL_DIR is intentionally configurable because a
large tmpfs is preferable to the small container overlay on Vast.ai.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


SOURCE_MODEL_ID = os.environ.get("SOURCE_MODEL_ID", "openai/gpt-oss-20b")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/dev/shm/gpt-oss-20b-bf16"))


def main() -> None:
    marker = MODEL_DIR / "config.json"
    if marker.exists() and (MODEL_DIR / "tokenizer.json").exists():
        print(f"Using existing dequantized model at {MODEL_DIR}", flush=True)
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {SOURCE_MODEL_ID} and dequantizing to {MODEL_DIR}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL_ID,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        use_cache=False,
        device_map="auto",
    )
    # Preserve the safe attention choice when VERL/vLLM reloads the saved config.
    model.config.attn_implementation = "eager"
    model.save_pretrained(MODEL_DIR, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(SOURCE_MODEL_ID).save_pretrained(MODEL_DIR)
    print(f"Wrote dequantized model to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
