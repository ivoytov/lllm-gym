# Pin the x86_64 vLLM image by digest so a rebuild cannot silently change its
# CUDA, PyTorch, or vLLM bits. Override VLLM_IMAGE deliberately when upgrading.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.25.1@sha256:2cc49b81319f7a66a33dd8bd63a7bfddae079122b33ce51989b6828a1f038c37
FROM ${VLLM_IMAGE}

ENV HF_HOME=/root/.cache/huggingface \
    VLLM_CACHE_ROOT=/root/.cache/vllm \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

EXPOSE 8000

# Model loading and compilation can take several minutes for production models.
# The long start period prevents an expected cold start from looking unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=5 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

STOPSIGNAL SIGTERM

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
     "--gpu-memory-utilization", "0.85", \
     "--max-num-seqs", "1", \
     "--max-num-batched-tokens", "2048", \
     "--enable-chunked-prefill", \
     "--fail-on-environ-validation", \
     "--shutdown-timeout", "120", \
     "--disable-uvicorn-access-log", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
