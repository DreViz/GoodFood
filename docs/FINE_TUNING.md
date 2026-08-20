# Fine-tuning the planner (Phase 8 — QLoRA)

Teach `qwen3:1.7b` the **procedure** it currently gets wrong, instead of paying
for it with parameters.

The measured problem (`reports/eval_summary.md`): the 1.7B passes **71.1%** of
the 45-conversation eval but only **44.4%** of booking, and its dominant failure
— 8 of its 13 failed conversations — is emitting `create_reservation` without
ever running `check_availability`. The 4B scores 93.3% but spills ~29% of its
layers onto the CPU on a 4 GB GPU. The question this phase answers: **can a
LoRA close the gap, and how much of it?**

Method: QLoRA — the base model stays frozen in 4-bit, ~35 M adapter parameters
train on top. Everything runs on the same RTX 3050 Laptop (4 GB).

---

## Pipeline at a glance

| # | Step | Command | Machine |
|---|---|---|---|
| 1 | Generate data | `python -m scripts.generate_training_data ...` | any |
| 2 | Train | `python -m scripts.train_planner_qlora ...` | GPU box |
| 3 | Export GGUF | `python -m scripts.export_planner_gguf` | GPU box |
| 4 | Register | `ollama create goodfoods-planner -f Modelfile.goodfoods-planner` | GPU box |
| 5 | Evaluate | `python -m scripts.evaluate --model goodfoods-planner` | GPU box |

No agent runtime code changes anywhere in this pipeline. Switching the planner
over is one environment variable.

---

## 0. Environment (training machine only)

```bash
python -m venv .venv-ft && .venv-ft\Scripts\activate        # keep it separate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-finetune.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

`requirements-finetune.txt` is deliberately not merged into `requirements.txt`:
it pulls ~3 GB and the API, agent, frontend and eval harness need none of it.
The CUDA wheel must be installed first, or pip silently gives you a CPU-only
torch. GGUF export additionally needs CMake + the Visual Studio C++ build tools
(Unsloth compiles llama.cpp on first use) — see the header of that file.

### Build artifacts and what to keep

`/data/`, `/adapters/` and `/gguf/` are gitignored. What each one is:

| Path | Size | Where it comes from | Rebuild cost |
|---|---:|---|---|
| `data/` | ~25 MB | generated locally by `generate_training_data.py` | 0.3 s (`--seed 42` is deterministic) |
| `adapters/planner_lora/` | ~35-70 MB | **the actual training output** — ~17 M LoRA parameters | 1.5-3 h of GPU time |
| `gguf/*.gguf` | ~1.1-1.4 GB | base weights downloaded from Hugging Face (`unsloth/Qwen3-1.7B`) + the adapter, merged and quantised | ~15 min from the adapter |

Only the **adapter** is expensive and irreplaceable — and it is small. To move
the fine-tune to another machine, copy `adapters/planner_lora/` (or push it to
the Hugging Face Hub) and re-run step 3 there; never commit the GGUF. The base
model is a public download, so the 1.4 GB file is 95% weights that anyone can
fetch in a couple of minutes.

## 1. Generate the training data

```bash
# smoke check (instant)
python -m scripts.generate_training_data --n 20 --out data/smoke.jsonl

# the real dataset
python -m scripts.generate_training_data \
    --n 3000 --out data/planner_train.jsonl --val-split 0.05 --seed 42
```

Writes `data/planner_train.jsonl`, `data/planner_train_val.jsonl` and
`data/manifest.json`. Same seed + same `--n` reproduces the files byte for byte.

Each line is one planner decision in chat format, structurally identical to what
`call_planner_llm` sends Ollama at inference time:

```json
{"messages": [
  {"role": "system",    "content": "<planner_prompt_phase2.txt, verbatim>"},
  {"role": "user",      "content": "7:30pm works\n\n--- Context (do not echo back) ---\nMemory: {...}\nRecent results: []\nCustomer profile: {}\nRespond ONLY with one valid JSON object."},
  {"role": "assistant", "content": "{\"plan\": \"execute\", \"action\": \"check_availability\", \"args\": {...}}"}],
 "meta": {"case": "a_time_completes", "phase": "availability", ...}}
```

`meta` is for the manifest and the tests — the trainer reads `messages` only.

Expected shape at `--n 3000 --seed 42` (verified against `manifest.json`):

| Split | | Group | | Plan | |
|---|---:|---|---:|---|---:|
| train | 2848 | availability | 50% | execute | 67% |
| val | 152 | booking | 20% | reply | 33% |
| | | discovery | 20% | | |
| | | edge | 10% | | |

By action: `check_availability` 43%, `search_restaurants_by_filters` 12%,
`get_seating_labels` 5%, `recommend_venues` 2%, **`create_reservation` 4.7%**.

The split is by unique user message, so the val count moves a little whenever
the template banks change — treat `manifest.json` as the source of truth.

### What the data teaches

Rules were derived from `planner_prompt_phase{1,2,3}.txt` and the deterministic
guards in `planner_agent.py` — never from the eval fixtures.

- **Discovery** — cuisine alone asks for more filters; cuisine + any second
  filter searches; a named restaurant goes straight to `get_seating_labels`;
  "near me" asks for a real area.
- **Availability** — **a stated time always means `check_availability`, never
  `create_reservation`.** This is the invariant the whole exercise exists for.
  Missing fields are collected date-first, then party size.
- **Booking** — no email means ask for the email; an email with everything else
  known means ask to confirm; only an explicit confirmation fires
  `create_reservation`; a decline fires nothing at all.
- **Edge guards** — party size 0/negative/over 50, an unparseable email, and an
  ambiguous "7pm or 8pm" all resolve to a re-ask, matching the Python guards so
  the model and the interceptors agree instead of fighting.

### Deliberate scope decisions

- **`cancel_reservation` / `modify_reservation` are not trained.** At runtime the
  entire manage flow is decided by pre-LLM interceptors in `planner_agent.py`
  (guards 1, 2, 4, 5) before the model is consulted — and both models already
  score 9/9 there. Training them could only introduce disagreement.
- **The booking group is 20%, not the 30% originally planned**, and
  `create_reservation` is only 4.7% of samples. Same reasoning taken one step
  further: guards 3 and 8 (`planner_agent.py:515` and `:635`) return
  `create_reservation` *without calling the model* whenever every field is in
  memory and the user gives a short affirmative, so budget spent on those turns
  changes nothing at runtime. Worse, over-representing "book it now" risks
  reinforcing the exact over-eagerness this fine-tune exists to remove. The
  confirmation case is kept but small — the guards do miss some phrasings
  ("that all looks right, go ahead" matches no `_CONFIRM_START` pattern), so the
  model still has to know how to build the call. The freed budget went to the
  availability phase, where the model genuinely decides and where the measured
  failure lives.
- **Dates stay in the user's words; times are normalised.** A newly stated date
  is emitted raw (`"next monday"`) because resolving it to ISO at generation
  time would bake the generation date into the weights — Python normalises it
  downstream, exactly as it does for the 4B today. A date already in memory is
  ISO and is copied verbatim. Times are calendar-independent, so `"7:30pm"` is
  emitted as `"19:30"`, matching both the Phase-2 prompt and memory.
- **`location_id` is never emitted.** `dispatch_tool` resolves it from the
  restaurant name, and `memory.location_id` is never populated at runtime.
- **One divergence from the Phase-2 prompt, on purpose.** That prompt still
  tells the planner to emit `create_reservation` when the user picks a slot.
  The Python layer overrides this (it asks for the email instead), and the
  eval scores the override. Training follows the enforced behaviour, not the
  stale prompt line.

---

## 2. Train

```bash
python -m scripts.train_planner_qlora --dry-run          # validate anywhere, no GPU

python -m scripts.train_planner_qlora \
    --data data/planner_train.jsonl \
    --max-seq-len 4096 --batch-size 1 --grad-accum 16
```

`--dry-run` loads and validates the dataset, resolves every hyperparameter and
exits 0 without importing torch, unsloth or CUDA. Run it before shipping the
repo to the GPU box.

**Use `--max-seq-len 4096`.** The default is the specified 2048, but the
Phase-1 (discovery) system prompt alone is 7.8 KB, which puts discovery samples
at roughly 2.1k tokens — over the limit. Truncation removes the *end* of the
sequence, which is the assistant's JSON, i.e. the label. The trainer tokenises
the whole dataset before training and **aborts** if any sample would be
truncated, so this cannot silently corrupt a run. Availability and booking
samples sit around 1.2–1.4k tokens.

Defaults: `unsloth/Qwen3-1.7B` in 4-bit, LoRA r=16 / alpha=32 / dropout 0 on
`q,k,v,o,gate,up,down`, lr 2e-4 cosine, 20 warmup steps, 2 epochs, adamw_8bit,
gradient checkpointing on, packing off. Loss is masked to assistant tokens via
`train_on_responses_only` — the 4 KB system prompt contributes no gradient.

Outputs:

```
adapters/planner_lora/        LoRA adapter + tokenizer
adapters/training_log.json    [{step, train_loss, val_loss, lr}, ...]
adapters/train_config.json    resolved hyperparameters
adapters/sample_rendered.txt  first example after chat templating
```

**Runtime estimate (RTX 3050 Laptop, 4 GB):** ~375 optimizer steps for 2 epochs
over 3k samples at effective batch 16 — expect roughly **1.5–3 hours**. Smoke
it first with `--max-steps 20` (a few minutes) and confirm the loss moves and
`sample_rendered.txt` looks right before committing to the full run. Replace
this estimate with the measured number once the first run lands.

If VRAM runs out: `--batch-size 1 --grad-accum 32`, then `--max-seq-len 3072`.

---

## 3. Export to GGUF

```bash
python -m scripts.export_planner_gguf --dry-run   # checks inputs, no ML imports
python -m scripts.export_planner_gguf
```

Merges the adapter into the base and quantises to **Q4_K_M** — the project's
quantization floor (`docs/model_specs.md`) — producing
`gguf/goodfoods-planner-q4km.gguf` (~1.4 GB, same class as the stock 1.7B).
Needs ~10 GB of free disk for the intermediate merge, which is deleted
afterwards unless `--keep-work-dir` is passed.

---

## 4. Register with Ollama

```bash
ollama create goodfoods-planner -f Modelfile.goodfoods-planner
ollama list | grep goodfoods-planner
```

`Modelfile.goodfoods-planner` pins temperature 0.6 / top_k 20 / top_p 0.95 /
num_ctx 8192 and a ChatML template. Those `PARAMETER` lines matter: the agent
sends no `options` block, so whatever the Modelfile declares *is* the runtime
sampling configuration.

Point the agent at it:

```bash
# .env
OLLAMA_MODEL=goodfoods-planner
```

---

## 5. Evaluate

```bash
# the 45-conversation benchmark — directly comparable to the 1.7B and 4B runs
python -m scripts.evaluate --model goodfoods-planner

# the 10 held-out conversations the model has never seen in any form
python -m scripts.evaluate --model goodfoods-planner \
    --fixture tests/eval/heldout_conversations.yaml
```

Reports land in `reports/eval_goodfoods-planner_<timestamp>.{json,md}`.

Baselines to beat, from `reports/eval_summary.md`:

| | qwen3:1.7b | qwen3:4b | fine-tuned 1.7B |
|---|---:|---:|---:|
| Conversations | 32/45 (71.1%) | 42/45 (93.3%) | _to measure_ |
| Booking (C) | 4/9 (44.4%) | 7/9 (77.8%) | _to measure_ |
| Edge (D) | 6/9 (66.7%) | 9/9 (100%) | _to measure_ |

The number that matters is **booking**, and specifically whether premature
`create_reservation` disappears. A fine-tuned 1.7B that matches the 4B on
booking while staying fully GPU-resident and 2.2× faster is the result worth
writing up; anything less should be reported as measured, including a negative
result.

Reverting costs one line: set `OLLAMA_MODEL` back to `qwen3:4b`.

---

## Integrity statement

**The training data is independent of the evaluation sets.**

- `scripts/generate_training_data.py` builds every utterance from template banks
  written from the phase prompts and the planner's guard logic. It never reads
  `tests/eval/conversations.yaml` or `tests/eval/heldout_conversations.yaml`;
  `tests/training/` asserts this statically (AST scan for any live reference)
  and dynamically (zero verbatim overlap across 3000 generated samples and both
  fixture sets).
- The only permitted overlaps are bare affirmatives — "yes", "book it",
  "go ahead" — which are enumerated in `planner_agent._strict_affirmatives`.
  They come from the agent's own code, and there is no other way to phrase a
  confirmation. Every such collision is listed explicitly in
  `ALLOWED_FIXTURE_COLLISIONS`; anything else fails the suite.
- The 45 fixtures are unmodified. The 10 held-out conversations were authored
  separately and share no turn with either the baseline set or the generator.
- Both eval sets are therefore genuine held-out benchmarks, and the reported
  numbers are comparable to the 1.7B and 4B runs on identical fixtures.

Run the guards any time:

```bash
pytest tests/training -q          # 45 tests, no GPU, no network, ~1s
```
