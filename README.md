# lab_scripts vLLM backend

This image is the local vLLM backend for `lab-rollouts`. The default model and
engine settings target one long sequence on an 8 GB NVIDIA GPU.

## Start the backend

Docker Compose supplies the runtime settings that cannot be encoded safely in
the Dockerfile: GPU access, automatic restart, an init process, shared memory,
rotated logs, loopback-only networking, and persistent model/compile caches.

Create the local secrets file before starting the service:

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` and set `HF_TOKEN` to a Hugging Face token with read access to the
model. Compose injects variables from this file into the vLLM container. The
real `.env` file is excluded from Git and from the Docker build context.

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f --tail=100 vllm
```

Wait until `docker compose ps` reports the service as healthy before starting
rollouts. The first start can spend several minutes downloading and compiling;
later starts reuse the named `huggingface-cache` and `vllm-cache` volumes.

Do not use `docker compose down -v` unless deleting both caches is intentional.

## Run rollouts

The default backend executes one sequence at a time, so match it with low client
concurrency and a long request timeout:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1

poetry run lab-rollouts \
    --model-name Qwen/Qwen3-0.6B \
    --stage-name example \
    --input-path /data/amelia-rlvr-queries.parquet \
    --output-path /data/predictions.parquet \
    --max-concurrency 1 \
    --request-timeout 3600
```

Keep the output Parquet and its `.checkpoint.sqlite3` journal on durable host
storage. Rerunning the same command resumes missing rollouts.

## Operational notes

- `restart: unless-stopped` restarts an exited server. Docker does not restart a
  process merely because its health status becomes `unhealthy`; monitoring or an
  orchestrator must act on that state.
- The API is published only on host loopback. If remote access is required, put
  an authenticated reverse proxy in front of it rather than publishing port 8000
  directly.
- Scrape `http://127.0.0.1:8000/metrics` and monitor NVIDIA Xid/ECC events,
  temperature, VRAM, host RAM, and free disk space.
- Consider enabling Docker daemon `live-restore` on a dedicated Linux host.
- The image is pinned to a vLLM v0.25.1 x86_64 digest. Update the tag and digest
  together after testing a new release.

To use a different model or engine configuration, replace the image command in
an environment-specific Compose override rather than editing the base service.
