# The official CUDA image keeps this container portable across supported NVIDIA
# GPUs. Override VLLM_VERSION at build time when upgrading vLLM.
ARG VLLM_VERSION=v0.25.1
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Keep `vllm serve` as the stable interface. `docker run` arguments replace CMD,
# so a larger model and H100-oriented settings can be supplied without rebuilding.
ENTRYPOINT ["vllm", "serve"]

# Proof-of-concept defaults for an 8 GB CUDA GPU:
# - Qwen3-0.6B is the smallest official Qwen3 dense model.
# - FP16 weights work on Turing-class GPUs; Qwen's block-FP8 weights do not.
# - FP8 KV cache leaves enough room for a single 40K-token sequence.
# - Chunked prefill bounds peak memory while CUDA graphs remain enabled.
CMD ["Qwen/Qwen3-0.6B", \
     "--dtype", "half", \
     "--max-model-len", "40960", \
     "--kv-cache-dtype", "fp8_e5m2", \
     "--gpu-memory-utilization", "0.90", \
     "--max-num-seqs", "1", \
     "--max-num-batched-tokens", "2048", \
     "--enable-chunked-prefill", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
