# Model Specifications (Phase 1 / WS1)

Records the local LLM setup and the **quantization floor** verification the plan
mandates (must be ≥ `Q4_K_M`). Reused by Phase 4 (quantization benchmark).

Served locally via **Ollama** (Docker container `goodfoods-ollama`, port `11434`).
Model tag is configurable through `OLLAMA_MODEL` in `.env` (see `app/config.py`).

## Verified models

| Model | Params | Quant | Ctx length | On-disk | Floor ≥ Q4_K_M | Role |
|---|---|---|---|---|---|---|
| `qwen3:4b` | 4.0B | **Q4_K_M** | 262144 | 2.5 GB | ✅ | dev / iteration (default) |
| `qwen3:8b` | 8.2B | _pending re-pull_ | — | — | ⏳ to verify | eval / benchmark (Phase 3/4) |

`qwen3:4b` verified via `ollama show qwen3:4b` — reports `quantization Q4_K_M`,
which meets the floor. **Do not pull tags below Q4_K_M.**

> `qwen3:8b` was pulled once but its manifest was lost when the Ollama container
> exited during setup (host suspend / daemon restart, not OOM). It must be
> re-pulled and its quant re-verified before Phase 3 eval:
> ```bash
> docker exec goodfoods-ollama ollama pull qwen3:8b
> docker exec goodfoods-ollama ollama show qwen3:8b | grep quantization
> ```

## Model capabilities (qwen3:4b)

`completion`, `tools`, `thinking`. qwen3 is a **reasoning model**: even with
reasoning disabled it emits a short preamble terminated by `</think>` before the
answer.

- Reasoning is disabled via the top-level `think: false` request parameter
  (`OLLAMA_THINK=false`). This keeps latency sane on CPU and yields clean output.
  The `/no_think` prompt switch was rejected — in testing it caused the model to
  ramble (900+ tokens) instead of suppressing reasoning.
- Any residual `</think>` preamble is stripped in code by
  `strip_model_reasoning()` (`app/agent/llm_utils.py`) before JSON/prose use.
- Note: the planner's old `"stop": ["}"]` was a **no-op** — Ollama only honours
  `stop` inside `options`, not at the request top level. It was removed.

## Host environment (observed)

| Resource | Value |
|---|---|
| GPU | none detected — **CPU-only inference** |
| CPU | 8 cores |
| RAM | ~15 GiB total |
| Throughput | `qwen3:4b` ≈ 6–7 tok/s (CPU) |

**Operational note:** RAM is tight (~0.6 GiB free with `qwen3:4b` loaded
alongside Postgres and other host containers). Do **not** load `qwen3:8b`
(~5–6 GiB) concurrently with `qwen3:4b` — run one model at a time to avoid the
Ollama container being killed.

_Last verified: 2026-07-27._
