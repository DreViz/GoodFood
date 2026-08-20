# Model Specifications

Records the local LLM setup, quantization-floor verification (must be ≥
`Q4_K_M`), and the runtime measurements behind the 1.7B-vs-4B comparison
(`reports/eval_summary.md`). Model tag is configurable via `OLLAMA_MODEL`
(`app/config.py`).

## Verified models

| Model | Params (actual) | Quant | Native ctx | On-disk | Floor ≥ Q4_K_M | Role |
|---|---|---|---|---|---|---|
| `qwen3:1.7b` | 2.0B | **Q4_K_M** | 40960 | 1.4 GB | ✅ | comparison lower-bound (Phase 4) |
| `qwen3:4b` | 4.0B | **Q4_K_M** | 262144 | 2.5 GB | ✅ | **selected** — dev + eval default |
| ~~`qwen3:8b`~~ | 8.2B | Q4_K_M | — | 5.2 GB | n/a | **dropped 2026-08-10** — see note |

Note the tag-vs-params discrepancy: the `qwen3:1.7b` tag reports **2.0B**
parameters via `ollama show`. The 1.7/4 naming refers to the model series, not
an exact count — worth stating plainly if asked.

**Why 8B was dropped** (2026-08-10): at ~5 GB runtime it exceeds the 4 GB VRAM
budget; partial CPU offload made sustained inference thermally unsustainable on
this laptop during the comparison run. The 1.7B-vs-4B pivot gives a wider
quality/cost span (2.0B → 4.0B actual params) on the same hardware with no
thermal confound for the smaller model.

## Runtime footprint (measured 2026-08-17, Ollama num_ctx=8192)

| Model | Runtime residency | GPU placement | VRAM used |
|---|---|---|---|
| `qwen3:1.7b` | 2.4 GB | **100% GPU** | 2353 MiB / 4096 MiB |
| `qwen3:4b` | 4.2 GB | **71% GPU / 29% CPU** | 3061 MiB / 4096 MiB |

**The 4 GB VRAM budget holds the 4B weights (2.5 GB) but not weights + KV
cache (4.2 GB at 8K context)** — Ollama silently offloads ~29% of the layers
to CPU. The 1.7B fits entirely on GPU. This placement difference is part of
the measured throughput gap, and it is disclosed rather than hidden: the 4B
numbers on this host are what a 4 GB card actually delivers.

## Throughput (warm, GPU-resident, `think:false`)

| Model | tok/s | Sample |
|---|---|---|
| `qwen3:1.7b` | **48.9** | 125 tokens |
| `qwen3:4b` | **22.3** | 744 tokens |

Methodology: `POST /api/chat`, single non-streaming generation, one warm-up
call before timing; `eval_duration`/`eval_count` from the Ollama response.
Competing model evicted (`ollama stop`) before each measurement so residency
is exclusive. ~2.2× ratio is consistent with the parameter delta plus the
4B's partial CPU offload.

## Model capabilities (qwen3 family)

`completion`, `tools`, `thinking`. qwen3 is a **reasoning model**: even with
reasoning disabled it emits a short preamble terminated by `</think>` before
the answer.

- Reasoning disabled via the top-level `think: false` request parameter
  (`OLLAMA_THINK=false`). The `/no_think` prompt switch was rejected — it
  caused the model to ramble (900+ tokens) instead of suppressing reasoning.
- Residual `</think>` preamble is stripped by `strip_model_reasoning()`
  (`app/agent/llm_utils.py`) before JSON/prose use.
- The planner's old `"stop": ["}"]` was a **no-op** — Ollama only honours
  `stop` inside `options`. Removed.

## Host environment

| Resource | Value |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop, **4 GB VRAM** |
| CPU | AMD Ryzen 9 5900HS |
| RAM | 16 GB (15.4 usable) |
| Runtime | Native Ollama on Windows, port 11434 |

Historical note: an earlier CPU-only setup (Ollama in Docker, ~6–7 tok/s on
4B) was retired; all numbers on this page are from the native Windows +
GPU setup observed 2026-08-17.

_Last verified: 2026-08-17._
