# FLUX.1-dev: Gradio + vLLM-Omni

```
           ┌─ fetch (one-shot, has HF token) ──► models volume (read-only)
           │                                          │
browser ─► Gradio :7860 ─► nginx LB :8091 ─┬─► vllm-gpu0 (offline) ◄┘
                                           └─► vllm-gpu1 (offline) ...
```

> **License:** FLUX.1-dev weights are **non-commercial**. To use them commercially, buy a license
> from Black Forest Labs (bfl.ai), or use FLUX.1-schnell, which is Apache-2.0 and drop-in here.

## Run
```bash
printf '%s' 'hf_xxx' > secrets/hf_token   # fine-grained, read-only token (see below)
docker compose up --build
```
`fetch` downloads the weights once (about 24 GB) and exits. The vLLM replicas start offline from the volume.
Open http://localhost:7860. The OpenAI-compatible API is at http://localhost:8091.

## Scaling
Diffusion requests are compute-bound and take seconds each, so it works differently from LLM serving:

| goal | how |
|---|---|
| **More throughput** | One replica per GPU (FLUX.1-dev in bf16 fits on a single 40/80 GB GPU), behind the `least_conn` LB. Throughput grows about linearly with replicas. To add a GPU, copy a `vllm-gpuN` block and add it to `nginx.conf`. |
| **Lower latency per image** | Set `ULYSSES_DEGREE=2` or `4` on a replica and give it that many GPUs. This is sequence parallelism, which splits one image's tokens across GPUs. It costs more GPUs per image, so use it only when latency matters more than cost. |
| CFG-parallel | Not useful here. FLUX.1-dev is guidance-distilled, so there is no second CFG pass to parallelize. |
| Multiple nodes | Move to Kubernetes: one pod per GPU (or per Ulysses group), with the NVIDIA GPU Operator and KEDA autoscaling on the queue depth or in-flight request count. Use readiness probes because cold start is minutes long. |
| Cheaper per image | FP8 quantization and step caching (TeaCache or cache-dit) cut cost but change output slightly. Benchmark them first. |

Gradio is only a UI. Its queue is per-process, so if you run several Gradio replicas, use sticky sessions.
High-volume clients should call the LB's `/v1/images/generations` directly.

## Hugging Face token at commercial scale
Production serving nodes should never hold an HF token.

1. **Sync the weights once, then serve offline.** Only a CI job or the `fetch` job uses the token. It
   downloads a **pinned commit SHA** (`MODEL_REVISION`) into storage you control: an S3/GCS bucket, a
   shared PVC, or an OCI artifact in your registry. Replicas run with `HF_HUB_OFFLINE=1`. This removes any
   runtime dependency on the hub (rate limits, outages, upstream changes) and keeps the token off every node.
2. **Token owner:** use an HF **organization** service or bot account that has accepted the gated license,
   not a personal account. Use a **fine-grained, read-only** token scoped to this one repo.
3. **Storage:** keep the token in a secret manager (Vault, AWS/GCP/Azure Secret Manager). In Kubernetes,
   inject it with External Secrets Operator or the Secrets Store CSI driver, and only into the sync job.
   Never put it in images, `.env` files in git, or `ENV`/`ARG`. If a build needs it, use
   `RUN --mount=type=secret`.
4. **Pass it as a file:** pass it as a file (`HF_TOKEN_PATH=/run/secrets/...`) instead of an env var, so it
   doesn't show up in `docker inspect`, crash dumps or child processes.
5. **Operations:** rotate on a schedule, and revoke immediately if it leaks. HF secret scanning helps, but
   don't rely on it. For large fleets, HF Enterprise adds org SSO, token policies and audit logs.

`secrets/hf_token` here is the local-dev stand-in for step 3.
