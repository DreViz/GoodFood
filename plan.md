# GoodFoods AI — Portfolio Upgrade Plan

> Status: **ACTIVE — implementation in progress.** Original draft approved
> 2026-07-24; Phases 1, 2, 5, 3, and 4 are complete. Phase 6 (README) is
> active. Phase 4 was **pivoted from 4B-vs-8B to 1.7B-vs-4B** (see §1) and
> completed 2026-08-17. Approach: extend the existing codebase, not rewrite.
>
> **Decisions settled 2026-07-24 (all shipped):**
> - **Frontend:** Next.js 14 (App Router) + Tailwind + shadcn/ui + Vercel AI SDK. ✅
> - **SSE protocol (Phase 2 task 9):** Option X — AI SDK data-stream protocol. ✅
> - **Cancel/modify tools (Phase 2 task 4 / B5):** Option B — implement both. ✅
> - **Sequencing:** Demo-first — Phase 1 → 2 → 5 → 3 → 4 → 6 → 7.
>
> **Re-decisions settled 2026-08-10:**
> - **Model comparison pivot:** Phase 4 compares **1.7B vs 4B**, not 4B vs 8B.
>   8B overheated the 4 GB laptop GPU and required partial CPU offload; 1.7B
>   is the cleaner lower-bound comparison and gives a wider quality/cost range
>   on the same hardware. See §1 (Hardware & Model Policy) for the full rationale.
> - **Semantic search (bonus workstream, not in original plan):** HuggingFace
>   embedding retrieval (`all-MiniLM-L6-v2`) integrated into
>   `search_restaurants` + `recommend_venues` as hybrid retrieval (SQL hard
>   filters + cosine soft-rank). Spec in `SEMANTIC_SEARCH_PLAN.md`; built and
>   shipping with the current eval.
>
> **Decision added 2026-08-17:**
> - **Phase 8 (QLoRA fine-tuning):** apply QLoRA (Unsloth, 4-bit NF4 + LoRA
>   adapters) to `qwen3:1.7b` on synthetic planner trajectories, targeting the
>   failure Phase 4 exposed (premature `create_reservation`; booking bucket
>   44.4%). Runs after Phase 6/7. Hard integrity constraint: training data is
>   generated independently of the 45 eval fixtures; the fixtures remain the
>   untouched benchmark. See Phase 8.

---

## Current status (2026-08-10)

| Phase | Status | Result |
|---|---|---|
| 1 — Local model reconnect | ✅ done | qwen3:4b on GPU (native Ollama, ~25 tok/s) |
| 2 — API cleanup + cancel/modify | ✅ done | B1–B8 bugs fixed; SSE streaming live |
| 5 — Next.js frontend | ✅ done | `localhost:3000` chat UI + tool-trace panel |
| 3 — Eval harness | ✅ done | 45 cases; **qwen3:4b = 42/45 (93.3%)** after bug-fix pass |
| (bonus) Semantic search | ✅ done | Hybrid retrieval, in-memory embedding index |
| 4 — Model comparison | ✅ done | 1.7B **71.1%** vs 4B **93.3%** → 4B selected; `reports/eval_summary.md` |
| **6 — README** | 🔄 active | README + CLAUDE.md + docs rewritten & committed; UI screenshots + video pending |
| 7 — Deploy | ⬜ later | — |
| **8 — QLoRA fine-tune** | 🔄 **active** | Task 1 ✅ env gate passed (10-step train, peak 1.66 GiB/4 GiB); pipeline code being generated on second machine (branch `phase8-qlora`) |

**4B eval headline:** started at **33%** (broken date normalization keystone +
cancel/modify interceptor gaps), fixed in a targeted pass to **93.3%**. The 3
remaining failures: 1 minor real bug (B08 email-collection gap in Phase 2→3
transition), 2 fixture/data issues (C06 over-strict expectation, C09 seed-data
mismatch where 6:30pm is genuinely booked). See `EVAL_BUGS.md` (historical) and
`reports/eval_qwen3_4b_20260810T071636Z.md` (latest).

**Phase 4 result (2026-08-17):** `qwen3:1.7b` = **32/45 (71.1%)**; booking
bucket collapsed to **44.4%** — dominant failure is premature
`create_reservation` (skips `check_availability`) in 8 of 13 failed
conversations. Cancel/modify = 9/9 on both models (interceptor-guarded paths
are model-independent). **Decision: 4B is the floor.** Full comparison in
`reports/eval_summary.md`; specs in `docs/model_specs.md`.

---

## 0. Audit Recap (what this plan is based on)

### Architecture already in place (keep)
- Two-agent split: **Planner** (`app/agent/planner_agent.py`) emits strict JSON;
  **Responder** (`app/agent/responder_agent.py`) turns tool output into user text.
- Orchestrator: `app/agent/agent.py` (`process_user_query`).
- Three-phase prompt system: `planner_prompt_phase1/2/3.txt`, selected by
  `memory.state["phase"]` via `load_prompt_for_phase()`.
- Hardened planner: JSON repair, schema validation, field cleaners
  (`safe_extract_*`, `strip_memory_if_unsupported`), and Python-level
  interceptors (early email capture, premature-booking block).
- Tool layer: `app/agent/tool_calls.py` with `TOOL_SPEC`, `TOOL_FUNCTIONS`,
  `dispatch_tool` (name normalization + fuzzy restaurant→`location_id`).
- Slot manager: `app/api/utils/slot_manager.py` (weekday-aware opening hours,
  capacity checks).
- Persistence: SQLAlchemy + Postgres JSONB (`app/data/db_models.py`).
- FastAPI app at `app/api/main.py`; Streamlit thin client at `app/main.py`
  (**to be replaced** — see Phase 2 & Phase 5).

### Broken / blocking issues (fix early)
| # | Issue | Location |
|---|---|---|
| B1 | Model + endpoint hardcoded; old `llama3.2:3b` deleted → silent fallback to mock greeting | `planner_agent.py:33-34`, `responder_agent.py:10-11` |
| B2 | Double prefix on routers (`/reservations/reservations/`, etc.) | `routes/{reservations,availability,restaurants}.py` declare `prefix=` *and* `main.py` mounts with `prefix=` |
| B3 | `routes/reservations.py:83` passes `restaurant.id` (PK) where `slot_manager` expects `location_id` → REST booking endpoint never validates availability | `routes/reservations.py:83` |
| B4 | Agent-path email send uses wrong kwarg names (`to_email=`, `body_html=`) vs signature `(recipient, subject, body)` → `TypeError` swallowed, `email_sent` always False | `tool_calls.py:382-393` vs `email_service.py:15` |
| B5 | Phase 3 prompt + planner `allowed_actions` reference `cancel_reservation` / `modify_reservation`; neither is in `TOOL_SPEC` / `TOOL_FUNCTIONS` | `planner_prompt_phase3.txt`, `planner_agent.py:91-101`, `tool_calls.py:439` |
| B6 | `/agent/memory/update` calls non-existent `ConversationMemory.update_from_user` | `main.py:75` |
| B7 | `tests/` are 0-byte placeholder files | `tests/*.py` |
| B8 | Streamlit renders user input via `unsafe_allow_html=True` | `app/main.py:229` (becomes moot when Streamlit is removed) |

### Missing vs. workstreams
WS1: no config indirection for model; no smoke test. WS2: agent logic is already
service-layer, but HTTP surface is buggy and schemaless. WS3/WS4: no eval or
benchmark infrastructure. WS5: README exists but not recruiter-graded. WS6: no
containerization; **free-tier cloud cannot host the model (no GPU)** — see §6.

### Repo shape after this plan
```
/
├── app/                    # FastAPI backend (Phase 2 cleanup)
├── frontend/               # Next.js 14 app (NEW, Phase 5)
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── package.json
├── scripts/                # smoke_test, evaluate, benchmark
├── tests/                  # eval fixtures + scorer unit tests
├── reports/                # eval_summary.md, quant_benchmark.md
├── docs/                   # model_specs.md, deployment.md
├── Dockerfile              # API container (Phase 7)
├── docker-compose.yml      # api + db (Phase 7)
└── plan.md                 # this file
```

---

## 1. Hardware & Model Policy (binding for all phases)

| Resource | Value |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop, **4 GB VRAM** |
| RAM | 16 GB (15.4 usable) |
| CPU | AMD Ryzen 9 5900HS |

**Models:**
- `qwen3:4b` — **primary eval + dev model.** Fits in VRAM (~2.5 GB at Q4_K_M),
  runs fully on GPU at ~25 tok/s. Current eval baseline: **42/45 (93.3%)**.
- `qwen3:1.7b` — **comparison point (added 2026-08-10, replaces 8B).** Tiny
  (~1 GB at Q4_K_M), fits comfortably in VRAM, expected to be the lower-bound
  quality test.
- ~~`qwen3:8b`~~ — **dropped 2026-08-10.** At ~5 GB Q4_K_M it exceeds the 4 GB
  VRAM budget; partial CPU offload made inference thermally unsustainable on
  the laptop (overheated during the comparison run). 8B-vs-4B would also have
  been a narrower quality band than 1.7B-vs-4B — the pivot gives a wider,
  more informative cost/quality range on the same hardware.

**Why the 1.7B-vs-4B comparison is the better story (interview framing):**
- Both models fit entirely in VRAM → clean apples-to-apples GPU inference, no
  CPU-offload confound muddying the latency numbers.
- 4× parameter span (1.7B → 4B) reveals the quality/cost slope clearly.
- Answers a real engineering question: *"how small can we go before quality
  drops below an acceptable threshold?"* — the actual day-job AI-eng tradeoff.

**Quantization floor: Q4_K_M.** Never go below. Phase 1 verified that the
default tags `qwen3:4b` / `qwen3:1.7b` resolve to ≥ Q4_K_M.

**Config policy:** model name, Ollama endpoint, DB URL, and SMTP creds all read
from environment via a single `app/config.py` (Pydantic `BaseSettings`). No
hardcoded values in agent code. `.env.example` documents every variable.

**Frontend compute footprint:** the Next.js app runs in the browser and on
Vercel's edge — **zero impact on your 4 GB VRAM budget**. The LLM never runs on
the frontend. Only the FastAPI backend talks to Ollama, exactly as today.

**Deployment honesty:** free-tier cloud (AWS EC2 free tier, GCP e2-medium,
Render free, Railway free) has **no GPU**. Hosting qwen3:4b there means full
CPU inference at ~1–3 tok/s — unusable for a demo. The plan therefore treats
the live-cloud-hosted-LLM path as **out of scope** and substitutes a
"Vercel-hosted UI + Render-hosted API + bring-your-own-local-Ollama + recorded
demo" path (see Phase 7).

---

## 2. Prioritization (impact-per-hour)

| Rank | Phase | Why | Est. |
|---|---|---|---|
| 1 | **Phase 1 (WS1)** | Unblocks every other phase. Without a live model, nothing can be tested or demoed. Cheapest win. | 2–4 h |
| 2 | **Phase 3 (WS3 — Eval)** | Strongest resume differentiator. A candidate who shows "45-conversation eval harness with tool-call accuracy and a 1.7B-vs-4B model-size comparison" stands out. | 10–14 h |
| 3 | **Phase 4 (WS4 — Quant)** | Pairs with Phase 3 (reuses its prompts); produces the comparison table that goes on the resume. | 3–5 h |
| 4 | **Phase 5 (Next.js UI)** | Replaces Streamlit. Polished demo UI, recruiter-recognized stack, live URL via Vercel. | 6–10 h |
| 5 | **Phase 6 (WS5 — README)** | Cheap once Phase 3/4/5 results exist; recruiter-facing. | 3–4 h |
| 6 | **Phase 2 (WS2 — API)** | Agent logic already service-layer; real work is bug fixes (B2/B3/B4) + Pydantic schemas + CORS + SSE protocol for the new frontend. Done before Phase 5 so the UI hits a clean API. | 5–7 h |
| 7 | **Phase 7 (WS6 — Deploy)** | Lowest resume ROI. Honest scoping: containerize API + Vercel UI + recorded demo, not live LLM hosting. | 5–7 h |

> **Execution order is NOT priority order.** Active sequence (Demo-first):
> **Phase 1 → 2 → 5 → 3 → 4 → 6 → 7.** Phase 2 must precede Phase 5 (frontend
> needs a clean API + correct SSE protocol). Phase 3 must precede Phase 6
> (README quotes eval results). Phase 5 must precede Phase 7 (deploy needs the
> UI to exist). Demo-first puts UI before eval — trades a little interview
> defensibility (eval results come later) for an earlier polished demo video.
>
> **If you can only do one phase this week: Phase 1.** It restores the demo,
> produces a recordable interaction, and unblocks all later work.

### Sequencing call-out: Phase 5 (UI) vs Phase 3 (Eval)
**Decision: Demo-first** (1 → 2 → 5 → 3 → 4 → 6 → 7). Reasoning recorded for
future reference: the user is job-hunting alongside a full-time job and wants
a polished demo video available for early applications, accepting that eval
results come a few weeks later. If circumstances change (e.g., a specific
interview requires eval data sooner), re-litigate the ordering before starting
Phase 5.

---

## Phase 1 — Reconnect the local model (WS1)

**Goal:** the agent runs end-to-end on `qwen3:4b`, exercised by a smoke test.
**Exit criteria:** `python scripts/smoke_test.py` prints a green trail for one
search → availability → booking conversation against `qwen3:4b`.
**Estimate:** 2–4 hours.

### Tasks
1. **Verify Ollama runtime.** `ollama serve` up on `127.0.0.1:11434`.
2. **Pull models.** `ollama pull qwen3:4b` and `ollama pull qwen3:8b`.
3. **Verify quantization floor.**
   - `ollama show qwen3:4b` and `ollama show qwen3:8b`.
   - Confirm the `model.quantize_version` / fileinfo reports **Q4_K_M or above**
     for both. Record the exact quant in `docs/model_specs.md` for Phase 4 reuse.
   - If any tag resolves below Q4_K_M, stop and surface the finding before
     proceeding.
4. **Add `app/config.py`** (Pydantic `BaseSettings`):
   - `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)
   - `OLLAMA_MODEL` (default `qwen3:4b`) — single source of truth
   - `DATABASE_URL`, `GOODFOODS_EMAIL`, `GOODFOODS_EMAIL_PASSWORD`
   - `API_CORS_ORIGINS` (comma-separated; defaults to
     `http://localhost:3000,http://localhost:8501` — 8501 stays during
     Streamlit's deprecation window)
   - Load `.env` automatically.
5. **Refactor LLM call sites** to read from config:
   - `planner_agent.py:33-34` → import `get_settings()`.
   - `responder_agent.py:10-11` → same.
   - Both call `{base_url}/api/generate` and pass `settings.ollama_model`.
6. **Add `.env.example`** with all keys + documented defaults.
7. **Update `CLAUDE.md`** Key Configuration section to reflect env-var names.
8. **Write `scripts/smoke_test.py`** — no test framework, just a runnable script:
   - Resets conversation memory.
   - Sends three canned user turns through `process_user_query`:
     `"Any Italian spots in Bandra?"` → `"GoodFoods Bistro"` (or whichever the
     seed data returns) → `"tomorrow at 7:30pm for 4, email is
     smoke@test.com"`.
   - Asserts at each turn: planner JSON valid; correct `action` emitted; final
     turn yields `create_reservation` with `ok: True`.
   - Prints a clear PASS/FAIL trail. Exits non-zero on failure.
   - Configurable via `--model` flag so the same script works for 4B and 8B.

### Approval gate
Show me the smoke test output for `qwen3:4b` (and the quantization confirmation
for both models) before starting Phase 2.

---

## Phase 2 — API-first cleanup (WS2, backend only)

**Goal:** a clean, schema-validated HTTP surface for the agent + new frontend,
with the bugs from the audit fixed, cancel/modify implemented, CORS configured,
and the SSE protocol aligned with the Vercel AI SDK data-stream format.
**Exit criteria:** all routes respond at their *intended* paths; Pydantic
schemas validate every agent/booking request; `routes/reservations.py` POST
actually validates availability; `cancel_reservation` and `modify_reservation`
work end-to-end with email; `/agent/chat/stream` emits the AI SDK data-stream
protocol; CORS allows `localhost:3000`.
**Estimate:** 7–10 hours (up from 5–7 h to cover Option B cancel/modify work).

### Decision carried over: keep Streamlit during this phase?
**Deprecate, don't delete yet.** Streamlit stays runnable through Phase 4 as a
fallback / regression check. Deletion happens in Phase 5 once the Next.js UI
covers the same flows. This keeps git history clean and gives you a working UI
the whole time.

### Tasks
1. **Fix double-prefix (B2).** Remove `prefix=` from each router declaration in
   `routes/{reservations,availability,restaurants}.py`; keep the mount-time
   `prefix=` in `main.py`. Verify paths become `/reservations/`, `/availability/`,
   `/restaurants/`.
2. **Fix slot-manager ID bug (B3).** `routes/reservations.py:83` — change
   `get_available_slots(restaurant.id, ...)` →
   `get_available_slots(restaurant.location_id, ...)`.
3. **Fix agent email kwargs (B4).** Align `tool_calls.create_reservation`
   (lines 382-393) with the real `send_email(recipient, subject, body)` signature.
4. **Resolve tool/prompt mismatch (B5).** **Decision: Option B — implement
   both tools.** They ship as real, tested features with eval coverage (Phase 3
   bucket E).
   - Add to `TOOL_SPEC` and `TOOL_FUNCTIONS` in `tool_calls.py`:
     - `cancel_reservation(customer_email)` — finds the customer's most recent
       `status="confirmed"` reservation, soft-deletes it (sets
       `status="cancelled"`, never hard-delete), sends a cancellation email via
       the B4-fixed `send_email(recipient, subject, body)` signature. Returns
       `{ok: False, error: "No active reservation for <email>"}` if none.
     - `modify_reservation(customer_email, new_date?, new_time?, new_party_size?, new_seating_pref?)` —
       finds the most recent `status="confirmed"` reservation, re-validates
       slot availability for any new date/time/party_size (calls
       `get_available_slots`), updates only the supplied fields, sends a
       modification email. Returns errors for: no active reservation, requested
       new slot unavailable, invalid new party_size.
   - **"One active reservation per customer email"** = the most recent
     `status="confirmed"` row. Document this in `tool_calls.py` docstrings and
     the Phase 6 README so it's defensible in interviews.
   - Both tools reuse the existing responder path — the responder prompt
     already mentions them. No prompt rewrite needed beyond keeping the
     Phase 3 prompt references that are already there.
   - Adds ~2–3 h to Phase 2 vs Option A.
5. **Remove dead `/agent/memory/update` endpoint (B6)** and the unused
   `conversation_memory` instance in `main.py`. Memory flows through the planner
   instance only. Update CLAUDE.md.
6. **Add Pydantic schemas.** New file `app/api/schemas.py`:
   - `ChatRequest { message: str, session_id: str | None }`
   - `ChatResponse { reply: str, tool_output: dict | None, phase: str }`
   - `SearchRequest`, `AvailabilityRequest`, `BookingRequest` for the shortcut
     endpoints below.
7. **Replace raw `Request` parsing** in `routes/agent.py` with the `ChatRequest`
   schema. **Decide SSE format now** (see task 9).
8. **Add shortcut endpoints** (clean, documented in OpenAPI):
   - `POST /agent/chat` — non-streaming, returns full `ChatResponse`. This is
     what the eval harness (Phase 3) will call.
   - `POST /search` — wraps `search_restaurants` directly.
   - `POST /availability` — dedupe with the existing `/availability/`.
   - `POST /book` — wraps `create_reservation` directly, with full validation.
9. **SSE protocol for the Vercel AI SDK.** **Decision: Option X — adopt the
   AI SDK data-stream protocol on the backend.** Rewrite `/agent/chat/stream`
   to emit:
   - `0:"<token>"\n` for text parts (the responder's reply, tokenized on
     whitespace so the UI streams word-by-word as it does today).
   - `2:{...}\n` for data parts — emit a single JSON data part containing the
     planner `action`, `args`, tool `result`, and post-turn `phase`. This is
     what Phase 5's tool-trace panel renders without a second round-trip.
   - `d:[{"finishReason":"stop"}]\n` to terminate the stream cleanly.
   - Keep the non-streaming `POST /agent/chat` (Phase 2 task 8) for the eval
     harness — it returns the full `ChatResponse` in one shot.
   - Document the wire format in `docs/sse_protocol.md` so the choice is
     defensible in interviews.
10. **Add CORS middleware** to `main.py` using `settings.api_cors_origins`.
    Allow `OPTIONS` preflight. Verify from `localhost:3000` once Phase 5 starts.
11. **(Deferred) XSS hardening (B8).** Moot once Streamlit is removed in Phase 5.

### Approval gate
Show me the OpenAPI schema (`/docs`), a `curl` round-trip on `/agent/chat` and
`/book`, and your pick on the B5 (cancel/modify) and SSE-protocol decisions
before starting Phase 3.

---

## Phase 3 — Evaluation harness (WS3)

> **✅ COMPLETE 2026-08-10.** All 6 tasks shipped. 45 cases across 5 buckets
> (A–E) implemented in `tests/eval/conversations.yaml`; runner is
> `scripts/evaluate.py`; reports land in `reports/eval_<model>_<timestamp>.{json,md}`.
>
> **Headline result — `qwen3:4b`: 42/45 conversations (93.3%), 98/103 turns
> (95.2%).** By bucket: search 9/9, availability 8/9, booking 7/9, edge 9/9,
> cancel_modify 9/9. Started at 33% before a targeted bug-fix pass (date
> normalization keystone + cancel/modify interceptor gaps); see
> `reports/eval_qwen3_4b_20260810T071636Z.md` for the full per-case breakdown.
>
> **3 remaining failures** (triaged, deferred — none block Phase 4):
> - **B08** (real bug, minor): Phase 2→3 transition should ask "which email?"
>   when the user picks a slot but hasn't given an email; instead it jumps
>   straight to "Shall I confirm?". Fix is in the Phase-2 planner interceptor.
> - **C06** (fixture over-strict): expected `plan=reply` on a fresh-time input,
>   but the agent correctly ran `check_availability`. Fixture expectation needs
>   loosening, not code change.
> - **C09** (seed-data mismatch): fixture assumes 6:30pm at GoodFoods Rooftop
>   is available; the seed data has it genuinely booked. Either reseed or pick
>   a different time in the fixture.
>
> Section below preserved as the original design spec for reference.

**Goal:** a re-runnable `python evaluate.py` that exercises the agent on 30–50
multi-turn conversations and reports tool-call accuracy, slot-filling
correctness, and task-completion rate, separately for 4B and 8B.
**Exit criteria:** `reports/eval_qwen3-4b_<timestamp>.md` and
`eval_qwen3-8b_<timestamp>.md` exist with per-case pass/fail and aggregate
metrics. Numbers are honest (not cherry-picked).
**Estimate:** 10–14 hours. This is the most effort-intensive phase.

### Test set design (`tests/eval/conversations.yaml`)
45 conversations across five buckets, 9 each:
- **A. Search correctness** — single-turn and multi-turn discovery with varied
  filters (cuisine, zone, price, tag, ambiguous zone).
- **B. Availability correctness** — date in multiple formats ("tomorrow",
  "18 Nov", "18/11"), party size phrased varied ("table for two", "we are 4"),
  time-specified and time-unspecified flows.
- **C. Booking completion** — end-to-end happy path with email capture,
  seating preference, confirmation step.
- **D. Edge cases** — unavailable slot (should offer alternatives), invalid
  email format, party_size=0 or negative, ambiguous restaurant name,
  mid-conversation context switch ("actually, make it 6 people").
- **E. Cancel/modify flows** (new — exists because B5 = Option B) — cancel
  after booking (should call `cancel_reservation`, status flips to
  `cancelled`), modify time after booking (should call `modify_reservation`
  with new time + new slot validation), cancel with no active booking (should
  reply asking for clarification, not call the tool), modify with no active
  booking (same), modify to an unavailable slot (should detect via
  `get_available_slots` and offer alternatives instead of failing).

Each fixture:
```yaml
- id: A03
  category: search
  turns:
    - user: "Any Italian places in Bandra under 1000?"
      expect:
        planner_action: search_restaurants_by_filters
        args_subset: { cuisine: italian, zone: bandra, max_price: 1000 }
    - user: "GoodFoods Bistro"
      expect:
        planner_action: get_seating_labels
        memory_after: { phase: availability }
  expected_outcome: phase_reached_availability
```

### Runner (`scripts/evaluate.py`)
- Loads `conversations.yaml`, resets memory before each conversation.
- Calls `POST /agent/chat` (added in Phase 2) turn-by-turn.
- For each turn, compares actual planner output against `expect`:
  - **Tool-call accuracy:** right `action` + right args (subset match).
  - **Slot-filling correctness:** memory contains the right values after the
    turn (date normalized, party_size an int ≥ 1, etc.).
- At conversation end, scores **task completion**:
  - For category C: was a `create_reservation` call made with the expected
    field values, and did the DB actually get the row?
- Aggregates per-category and per-metric.
- CLI flags: `--model qwen3:4b|qwen3:8b`, `--categories A,B,C,D`,
  `--reset-db` (reseed between runs so booking state doesn't leak).
- Output: JSON (machine) + Markdown (human) report under `reports/`.

### Tasks
1. Author `tests/eval/conversations.yaml` — 40 cases, reviewed for realism.
2. Add `tests/eval/scorer.py` — pure functions, unit-testable.
3. Add `scripts/evaluate.py` — orchestration + reporting.
4. Run on `qwen3:4b`, capture report.
5. `--reset-db`, run on `qwen3:8b`, capture report.
6. **Write the honest comparison** into `reports/eval_summary.md`: 4B vs 8B
   tool-call accuracy, slot-filling, completion rate, and median per-turn
   latency. Do not hide 4B's weaknesses or oversell 8B.

### Approval gate
Review of `conversations.yaml` (before running) and of
`reports/eval_summary.md` (after). I will want to discuss any case where the
two models disagree.

---

## Phase 4 — Model comparison (1.7B vs 4B) (WS4)

> **Reframed 2026-08-10.** Originally "4B vs 8B quantization documentation";
> pivoted to **1.7B vs 4B** for the reasons in §1 (Hardware & Model Policy).
> The 8B path is dropped.

**Goal:** produce a 1.7B-vs-4B comparison table I can paste into README and
resume, measuring quality, speed, and memory on identical hardware.
**Exit criteria:** `reports/eval_qwen3_1.7b_<timestamp>.md` exists (full 45-case
eval on the 1.7B model), and `reports/eval_summary.md` contains the
side-by-side comparison table with honest commentary.
**Estimate:** 2–3 hours (down from 3–5 h — Phase 3 already built the harness).

### Important framing
**We already have the measurement instrument.** Phase 3's eval harness
(`scripts/evaluate.py` + `tests/eval/conversations.yaml`) runs 45 conversations
and reports per-bucket accuracy. Phase 4 just runs it on the second model and
compares. **No separate benchmark script is needed** — the eval *is* the
quality measurement, and it already exercises the same planner/tool/memory
stack a production deployment would use.

For speed/memory, we supplement with one-shot Ollama API timings (eval duration
+ eval count from the `/api/chat` response, plus `nvidia-smi` VRAM reads).

### Tasks
1. **Pull the model.** `ollama pull qwen3:1.7b`.
2. **Verify quantization floor.** `ollama show qwen3:1.7b` → confirm ≥ Q4_K_M.
   Record specs in `docs/model_specs.md` alongside the 4B entry.
3. **Run the full eval on 1.7B:**
   ```powershell
   python -m scripts.evaluate --model qwen3:1.7b
   ```
   Produces `reports/eval_qwen3_1.7b_<timestamp>.{json,md}`.
4. **Capture speed/memory for both models.** For each of `qwen3:1.7b` and
   `qwen3:4b`, run a warm-up call then a timed call via the Ollama API:
   ```bash
   curl -s --max-time 60 http://127.0.0.1:11434/api/chat \
     -d '{"model":"qwen3:1.7b","messages":[{"role":"user","content":"Write a sentence."}],"stream":false,"think":false}' \
     | python -c "import sys,json; d=json.load(sys.stdin); ms=d.get('eval_duration',0)//1000000; c=d.get('eval_count',0); print(f'{c/(ms/1000):.1f} tok/s' if ms else 'no data')"
   ```
   Capture VRAM with `nvidia-smi --query-gpu=memory.used`. Record in
   `docs/model_specs.md`.
5. **Write `reports/eval_summary.md`** — the honest comparison:

   | Metric | qwen3:1.7b | qwen3:4b |
   |---|---|---|
   | Conversations passed | __/45 | 42/45 |
   | Turns passed | __/103 | 98/103 |
   | Search (A) | __/9 | 9/9 |
   | Availability (B) | __/9 | 8/9 |
   | Booking (C) | __/9 | 7/9 |
   | Edge (D) | __/9 | 9/9 |
   | Cancel/modify (E) | __/9 | 9/9 |
   | Throughput (tok/s, GPU) | __ | ~25 |
   | VRAM at idle | __ | ~2.5 GB |
   | Model size on disk | ~1 GB | ~2.5 GB |

   Fill the blanks from the fresh run. **Do not hide 1.7B's weaknesses or
   oversell 4B.** The interesting finding is *where* 1.7B breaks — likely
   Phase-3 booking (multi-step slot filling) and edge cases (instruction
   following). That's the story.
6. **Interview framing for the summary.** Lead with the engineering question:
   *"how small can we go before quality drops below an acceptable threshold?"*
   The 1.7B-vs-4B answer is the differentiator. If 1.7B holds above ~80%, the
   cost story (half the VRAM, faster cold start) is real. If it collapses on
   booking, that's a defensible finding too — model-size selection matters.

### Approval gate
Review of `reports/eval_summary.md` before it goes into the README (Phase 6).

---

## Phase 5 — Next.js frontend (replaces Streamlit)

**Goal:** polished, recruiter-grade chat UI that consumes the FastAPI backend
via the Vercel AI SDK, deployable to Vercel for a live demo URL.
**Exit criteria:** `cd frontend && npm run dev` → `localhost:3000` renders a
chat interface that completes a full search → availability → booking flow
against `localhost:8000`; sidebar saves preferences; conversation reset works;
the build passes `npm run build` cleanly.
**Estimate:** 6–10 hours.

### Decision: delete Streamlit now?
Yes. At the end of this phase, `app/main.py` (Streamlit) gets moved to
`legacy/streamlit_app.py` for git-history reference and removed from the
recommended run path. README and CLAUDE.md updated.

### Tasks
1. **Scaffold** (in `frontend/`, separate from `app/`):
   ```bash
   npx create-next-app@latest frontend \
     --typescript --tailwind --app --no-src-dir --import-alias "@/*"
   cd frontend && npx shadcn@latest init
   ```
   Add shadcn components: `button`, `input`, `card`, `scroll-area`, `avatar`,
   `badge`, `separator`, `tooltip`.
2. **Install Vercel AI SDK:**
   ```bash
   npm install ai @ai-sdk/react zod
   ```
3. **Wire the chat** (Phase 2 chose Option X — AI SDK data-stream protocol):
   use the Vercel AI SDK's `useChat` hook directly against
   `http://localhost:8000/agent/chat/stream`. Read the `data` parts (planner
   `action`, `args`, tool `result`, `phase`) off the returned message metadata
   for the tool-trace panel (task 4).
4. **Build the chat UI:**
   - `app/page.tsx` — main page, layout (header card + chat area + sidebar).
   - `components/chat-messages.tsx` — message list with user/assistant bubbles
     using shadcn `Avatar` + `Card`.
   - `components/chat-input.tsx` — input box with send button; disable while
     assistant is responding.
   - `components/tool-trace.tsx` (optional but recommended) — a collapsible
     panel under each assistant message showing the planner JSON (`action`,
     `args`) and tool result. Great demo flourish, easy if Option X is chosen.
5. **Build the preferences sidebar:**
   - `components/preferences-sidebar.tsx`
   - Fields matching the current Streamlit sidebar: name, email, cuisine,
     max price, vibe tags (multi-select), quiet preference, date (next 7 days),
     seating preference.
   - On save → `POST http://localhost:8000/customers/profile` with the same
     payload shape Streamlit sends today.
6. **Conversation reset button** → `POST http://localhost:8000/agent/memory/reset`.
7. **Environment config:** `frontend/.env.local` with
   `NEXT_PUBLIC_API_URL=http://localhost:8000`. All fetches read from this.
8. **Loading & error states:**
   - Typing indicator (animated dots) during streaming.
   - Toast (shadcn `Sonner`) on API errors.
9. **Polish:**
   - Dark theme by default (matches current Streamlit dark aesthetic).
   - Mobile-responsive (chat stacks, sidebar collapses).
   - Branded colors (the `#4b6043` / `#D4A373` from the Streamlit CSS map
     cleanly to Tailwind tokens).
10. **`npm run build` passes.** Fix any TypeScript errors.
11. **Remove Streamlit:** move `app/main.py` → `legacy/streamlit_app.py`,
    remove Streamlit from `requirements.txt`, update `CLAUDE.md` Frontend
    section, update README run instructions.

### Approval gate
Working `localhost:3000` walkthrough (screenshots or short video) covering one
full search → availability → booking conversation, plus the preferences
sidebar. Review of the tool-trace panel if shipped.

---

## Phase 6 — Documentation (WS5)

**Goal:** recruiter-readable README with a clear problem statement first,
technical depth second. All claims defensible in an interview.
**Exit criteria:** `README.md` rewritten; architecture diagram present; eval +
quant results summarized inline with links to full reports; screenshots of the
new UI.
**Estimate:** 3–4 hours.

### README structure
1. **One-paragraph problem statement** (recruiter-readable).
2. **Demo** — link to live Vercel URL (post-Phase 7) + 60-sec video.
3. **Screenshots** — chat UI with tool-trace visible, sidebar preferences.
4. **Architecture diagram** — Mermaid, three boxes: Next.js (browser) → FastAPI
   (backend) → Ollama (local LLM), plus Postgres + SMTP. Show phase transitions
   in the agent layer.
5. **Tech stack** — one-liner per component (Next.js, FastAPI, SQLAlchemy,
   Ollama, qwen3, Postgres).
6. **Hardware & model setup** — the 4 GB VRAM constraint, why a
   1.7B-vs-4B comparison on this hardware, the Q4_K_M floor. This is the
   differentiator; lead with it.
7. **Quick start** — `ollama pull`, DB setup, run backend, run frontend (two
   terminals).
8. **Evaluation results** — table lifted from `reports/eval_summary.md` with a
   link to the full report.
9. **Quantization / model-size results** — table from `reports/eval_summary.md`
   (1.7B vs 4B, quality + speed + memory on the same 4 GB GPU).
10. **Roadmap** — short, honest list of what's not done.

### Tasks
1. Replace current `README.md` (feature-list style) with the structure above.
2. Generate Mermaid diagram, render-check it on GitHub.
3. Take UI screenshots after Phase 5.
4. Record 60-sec walkthrough video; link from README (can defer to Phase 7).
5. Update `CLAUDE.md` if any phase changed conventions.

### Approval gate
Full README review before Phase 7.

---

## Phase 7 — Deployment (WS6)

**Goal:** a live demo URL the user can put on a resume, plus a containerized
local-run path, without overclaiming.
**Exit criteria:** Next.js deployed to Vercel (live URL); FastAPI deployed to
Render free tier; `Dockerfile` + `docker-compose.yml` for full-stack local
runs; `docs/deployment.md` walks through all three paths with an explicit
"local Ollama required for chat" note.
**Estimate:** 5–7 hours.

### Honest scoping (read first)
**You cannot host qwen3 on a free-tier cloud instance.** Free tiers offer CPU
only; running a 4B model at ~1–3 tok/s is not a usable demo. The options:

| Path | Pros | Cons | Resume-defensible? |
|---|---|---|---|
| **A. Vercel UI + Render API + bring-your-own-Ollama + recorded demo (recommended)** | Free, live UI URL recruiters can browse, honest, runnable by anyone with a GPU laptop | Live URL's chat only works if visitor runs Ollama locally — recorded video carries the actual demo | **Yes** — "containerized API + Vercel UI + documented local-LLM path" |
| B. Cloud GPU instance (GCP A2 / AWS g4dn / RunPod) | True end-to-end live demo | $40–300/mo, not free tier | Yes, but expensive |
| C. Swap to hosted LLM API (GLM/GPT/Claude) | Cheap live demo | **Violates the local-model differentiator** — explicitly rejected | No |

**Recommendation: Path A.** The Vercel URL is real and browsable; the recorded
demo video shows the actual AI. If you want true end-to-end live chat later,
Path B is the upgrade.

### Tasks
1. **Vercel deploy (Next.js):**
   - Push repo to GitHub (if not already).
   - Import `frontend/` into Vercel. Set root directory to `frontend`.
   - Set env var `NEXT_PUBLIC_API_URL` to the Render URL (from step 2).
   - Deploy. Live URL like `goodfoods-demo.vercel.app`.
2. **Render deploy (FastAPI):**
   - Create `render.yaml` or use the web UI. Root directory = repo root.
   - Build command: `pip install -r requirements.txt`.
   - Start command: `uvicorn app.api.main:app --host 0.0.0.0 --port $PORT`.
   - Add a Postgres free-tier add-on; set `DATABASE_URL` env from it.
   - Set `OLLAMA_MODEL=qwen3:4b`, `OLLAMA_BASE_URL=http://127.0.0.1:11434`
     (placeholder — Render won't reach your local Ollama; documented in
     `docs/deployment.md`).
   - Note: SMTP env vars are optional for the deploy; email sends will
     silently no-op without them (already handled by B4 fix).
3. **`Dockerfile`** (multi-stage, for full-stack local runs): Python slim base,
   install requirements, copy app. Entry point `uvicorn app.api.main:app
   --host 0.0.0.0 --port 8000`.
4. **`docker-compose.yml`** — two services: `api` (build `.`) and `db` (postgres
   image with a volume). `DATABASE_URL` wired between them.
5. **`.dockerignore`** — exclude `frontend/node_modules`, `frontend/.next`,
   `.venv`, `.git`, `__pycache__`, `reports`.
6. **`docs/deployment.md`:**
   - **Path 1 — Local full stack:** Docker Compose for API+DB, run Ollama
     natively, run Next.js with `npm run dev`. Best for development.
   - **Path 2 — Deployed split:** Vercel UI + Render API. Explicit note that
     chat requires the visitor to run Ollama locally and point their browser
     at it — provide a one-line `ollama pull qwen3:4b` instruction. The live
     URL is browsable (UI loads, sidebar works) but chat won't respond without
     a reachable backend.
   - **Path 3 — Recorded demo (recommended for applications):** 60–90 sec
     video of the full flow. Host on YouTube unlisted or Loom.
   - **Path 4 (optional upgrade) — Cloud GPU:** RunPod / Lambda / GCP A2
     pointer, no detailed walkthrough.
7. **Record the demo video** (if not already done in Phase 5/6).

### Approval gate
Review of `docs/deployment.md` and the live Vercel URL before considering the
project done.

---

## Phase 8 — QLoRA fine-tuning (added 2026-08-17, runs after Phase 7)

**Status note (2026-08-17):** started ahead of Phase 7. Execution is split
across two machines:
- **Machine A (second laptop, larger token budget):** generates the pipeline
  CODE per a handoff spec — data generator, unit tests, training script,
  GGUF export, held-out fixtures, `docs/FINE_TUNING.md`. Branch:
  `phase8-qlora`. No training runs there.
- **Machine B (4 GB GPU laptop, this repo):** environment setup (task 1),
  then runs everything — data gen, training, export, ollama create, re-eval.
  The handoff spec pins the integrity rule (generator independent of the 45
  fixtures) and requires runtime formats be extracted from
  `planner_agent.py`, never invented.

**Goal:** raise the 1.7B planner's quality floor with QLoRA fine-tuning so the
cheap model becomes viable — or document how far fine-tuning moves the floor.
Either outcome is a defensible finding.
**Exit criteria:** `reports/eval_summary.md` gains a fine-tuned-1.7B row
measured on the SAME 45-fixture harness; the data generator + training config
are committed; numbers are honest (no fixture contamination).
**Estimate:** 10–20 h (environment friction is the wildcard).

### Framing — what we can claim
We **apply** QLoRA; Unsloth/PEFT/bitsandbytes implement it. The
resume-defensible claim is the pipeline around it: synthetic trajectory
generation from the state-machine spec, 4-bit + LoRA training on a 4 GB GPU,
GGUF/Ollama export, and before/after eval on an untouched benchmark.

### Hard integrity constraint (non-negotiable)
Training data is generated **synthetically** — a memory-state sampler +
utterance templates + paraphrase variation — and must be independent of
`tests/eval/conversations.yaml`. The 45 fixtures stay untouched as the
benchmark. Add ~10 held-out conversations as a second, never-trained-on
check. Training on the eval fixtures (or close paraphrases of them) would
contaminate every before/after number this project reports.

### Tasks
1. **Environment.** ✅ done 2026-08-17 (native Windows — no WSL2 needed).
   Dedicated `.venv-train`: torch 2.11.0+cu128, Unsloth 2026.8.18,
   bitsandbytes 0.50.1, peft 0.20.0, transformers 5.5.0 — pinned in
   `requirements-training.txt`. Gate passed via `scripts/smoke_qlora_env.py`:
   10 training steps on Qwen3-1.7B-4bit with LoRA r=16 attached (17.4M
   trainable params, 1.66%), loss 8.39 → 1.48, peak VRAM **1.66 GiB of
   4.0 GiB** — comfortable fit. Two bring-up fixes worth remembering:
   peft 0.20.0 has no `get_parameter_count()` (count `requires_grad`
   manually), and transformers 5.x refuses to train without explicit
   `labels` in the batch (no loss from `input_ids` alone).
2. **Synthetic data generator** (`scripts/generate_training_data.py`):
   - Sample conversation state: phase ∈ {1,2,3}, memory slots filled/missing,
     seeded restaurant.
   - Emit a user utterance from templates + paraphrase variation (the
     cuisine/zone/date/party phrasings are already catalogued in the phase
     prompts).
   - Emit ground-truth planner JSON using the same rules the interceptors
     encode (availability before booking, email required, etc.).
   - Output: chat-format triples (system = phase prompt, user = utterance +
     memory state, assistant = plan JSON), 1–5k examples, JSONL.
3. **Train.** `qwen3:1.7b`, 4-bit NF4, LoRA r=16 (attention + MLP
   projections), seq 2048, batch 1–2 + gradient accumulation, 1–3 epochs.
   Training targets formatted exactly as the planner parses them (post
   `think:false` strip, valid JSON).
4. **Export + serve.** Merge adapters → GGUF Q4_K_M → `ollama create
   goodfoods-planner` from a Modelfile. Same runtime path as every other
   model — no special-casing in the agent.
5. **Re-run the eval.** `python -m scripts.evaluate --model goodfoods-planner`.
   Three-way comparison: base-1.7B / tuned-1.7B / 4B. Also run the ~10
   held-out conversations and report them separately.
6. **Write up.** Extend `reports/eval_summary.md` (new row + commentary),
   patch the README model-comparison section, update this file's status
   table.

### Hardware notes
- 1.7B QLoRA fits the 4 GB VRAM (quantized base + adapter gradients +
  optimizer state). 4B QLoRA does not fit — 1.7B is the training target by
  necessity, which is also the story: *recover the cheap model*.
- Success bar: tuned-1.7B booking bucket materially above the base 44.4%
  (stretch: approach the 4B's 77.8%) while search/edge/cancel buckets hold.
  If held-out numbers lag fixture numbers, the write-up says so plainly.

### Risks
- bitsandbytes/Unsloth on native Windows — WSL2 was the fallback; **not
  needed**, the stack runs natively (task 1 gate).
- Overfit to template phrasing — mitigate with paraphrase variety + held-out
  conversations; disclose any fixture-vs-held-out gap.
- Output-format drift (think preamble, JSON quirks) — targets must match the
  exact parse path (`strip_model_reasoning` → JSON).

### Approval gate
Review of the generator's sample output + training config BEFORE the real
run; review of the three-way comparison table before the README patch.

---

- **4B may be too weak for Phase 2 edge cases.** Phase 3 surfaced this — and
  the 4B result was 93.3%, so the concern was largely unfounded. The remaining
  failures are documented in `EVAL_BUGS.md` and the Phase 3 annotation above.
- **1.7B may collapse on multi-step booking flows.** That's the hypothesis
  Phase 4 tests. If 1.7B's bucket-C (booking) accuracy drops sharply, the
  finding is real and defensible: model-size selection matters, and the
  cost/quality knee is measurable. Do not hide it. *(Update 2026-08-17: it
  did — booking 44.4%. See `reports/eval_summary.md`.)*
- **Phase 8 (QLoRA) could contaminate the benchmark if rushed.** The 45
  fixtures are the project's honesty anchor; training on them (or close
  paraphrases) invalidates every before/after number. Synthetic-only training
  data + held-out conversations are mandatory, and any fixture-vs-held-out
  gap gets disclosed in the write-up.
- **Phase 3 estimate (10–14 h) is the wildcard.** Writing 45 realistic
  multi-turn fixtures (up from 40 to cover cancel/modify bucket E) is the
  single biggest time sink in this plan. If time is tight, cut to 30 cases
  (still defensible) before cutting quality.
- **Cancel/modify scope (Phase 2 Option B).** Two new tools = more surface
  area for bugs (email-template issues, slot re-validation races, "which
  reservation does the customer mean?" ambiguity if they ever have multiple
  active). Mitigation: ship and eval `cancel_reservation` before
  `modify_reservation` — cancel is single-step, modify is multi-step and
  reuses cancel's resolution logic.
- **Next.js adds Node 20+ to the toolchain.** Two dev servers during
  development (Next on :3000, FastAPI on :8000). This is normal for full-stack
  work; document it in the README.
- **Two-process deploy (Vercel + Render) has cold-start latency.** Render's
  free tier sleeps after 15 min idle; first request takes ~30s to wake. The
  README should disclose this honestly.
- **Email flow depends on Gmail App Passwords.** Out of scope for portfolio
  correctness; document the setup, don't try to make it work in CI.
- **No CI/CD.** Adding GitHub Actions for lint + smoke test is a natural Phase
  8 (out of scope here).

---

## 4. What I need from you before starting Phase 1

Decisions settled 2026-07-24 (recorded in the header): SSE protocol = **Option
X**, B5 cancel/modify = **Option B (implement both)**, sequencing =
**Demo-first** (1 → 2 → 5 → 3 → 4 → 6 → 7). Remaining open items:

1. Approval of this plan (or edits).
2. Confirmation that Ollama is installed (or permission to walk through the
   install).
3. Anything you want me to *not* touch (e.g., keep the Streamlit CSS color
   palette in the Next.js port, or don't touch agent prompts beyond what's
   strictly required).

I will not begin Phase 1 until you sign off.
