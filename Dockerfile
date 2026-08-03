FROM python:3.12-slim-bookworm

ARG VLLM_VERSION=0.25.1
RUN pip install --no-cache-dir \
      "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl" \
      --extra-index-url https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
      libtcmalloc-minimal4 libnuma1 g++ curl \
    && rm -rf /var/lib/apt/lists/*

RUN printf '#!/bin/bash\n\
TCMALLOC=$(find /usr/lib /usr/local/lib -name "libtcmalloc_minimal.so*" | head -1)\n\
IOMP=$(find /usr/local/lib /usr/lib -name "libiomp5.so*" | head -1)\n\
export LD_PRELOAD="$TCMALLOC${IOMP:+:$IOMP}"\n\
exec "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh

ENV VLLM_CPU_KVCACHE_SPACE=2 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8000

RUN pip install --no-cache-dir \
      "trl>=0.16" datasets accelerate

ENTRYPOINT ["/entrypoint.sh"]
CMD ["vllm", "serve", "Qwen/Qwen3-1.7B", "--dtype", "bfloat16", "--max-model-len", "2048", \
     "--host", "0.0.0.0", "--port", "8000"]