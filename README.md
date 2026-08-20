# GoodFoods — AI Reservation Agent

A natural-language concierge for restaurant reservations. Users chat in plain
English to discover restaurants, check live availability, and book, modify, or
cancel a table — powered by a **two-agent LLM architecture** running on a
**local model** (Ollama · qwen3), a **FastAPI** backend, and a **Next.js** chat UI.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.139-009688?logo=fastapi&logoColor=white">
  <img alt="Next.js" src="https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-qwen3-000000?logo=ollama&logoColor=white">
  <img alt="Postgres" src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white">
</p>

---

## Why this project

Booking a table usually means forms, dropdowns, and dead ends. GoodFoods replaces
that with a conversation: the agent asks one question at a time, keeps track of
what you've said, and only calls a tool when it has enough information. Because
the model runs **locally**, there's no per-token API cost and no customer data
leaving the machine — a deliberate design constraint that shapes the whole system.

The interesting engineering question this project answers is **how small the
model can be** before a multi-step agent workflow breaks — measured, not
guessed. See [Model selection](#model-selection--the-4-gb-question) below.

## Demo

- 🖥️ **Live UI** — Vercel URL pending (Phase 7 deployment).
- 🎬 **60-sec walkthrough** — video pending, recorded with the Phase 7 deploy.

## Screenshots

_Pending capture: chat UI mid-booking with the tool-trace panel open, and the
preferences sidebar._

---

## Architecture

The core idea is a **separation of concerns between deciding and speaking**. A
*Planner* decides the next action as strict JSON; a *Responder* turns tool output
into human text. Everything talks to a local Ollama model and a PostgreSQL database.

```mermaid
flowchart LR
    User(("User"))

    subgraph Browser["Browser — Next.js 14"]
        UI["Chat UI<br/>useChat · shadcn/ui"]
        SB["Preferences<br/>sidebar"]
    end

    subgraph Backend["FastAPI backend · :8000"]
        direction TB
        RT["Routes<br/>/agent · /search · /book · ..."]
        ORCH["Agent orchestrator<br/>process_user_query"]
        MEM["Conversation memory<br/>phase + slots"]
        PLAN["Planner agent<br/>decides next action → JSON"]
        TOOLS["Tool layer<br/>dispatch_tool<br/>+ semantic retrieval"]
        RESP["Responder agent<br/>writes the reply"]
    end

    subgraph Services["Local services"]
        OLLAMA[["Ollama<br/>qwen3:4b"]]
        DB[("PostgreSQL<br/>JSONB")]
        MAIL[["SMTP · Gmail"]]
    end

    User --> UI
    UI -- "AI SDK data-stream (SSE)" --> RT
    SB -- "POST /customers/profile" --> RT
    RT --> ORCH
    ORCH <--> MEM
    ORCH --> PLAN --> OLLAMA
    ORCH --> TOOLS --> DB
    ORCH --> RESP --> OLLAMA
    TOOLS --> MAIL
```

### A single turn, end to end

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FE as Next.js (useChat)
    participant API as FastAPI /agent/chat/stream
    participant O as Orchestrator
    participant P as Planner (qwen3)
    participant T as Tools + Postgres
    participant R as Responder (qwen3)

    U->>FE: types a message
    FE->>API: POST { message }
    API->>O: process_user_query()
    O->>P: phase prompt + memory
    P-->>O: {"plan":"execute","action":"check_availability","args":{...}}
    O->>T: dispatch_tool(action, args)
    T-->>O: { ok, available_slots, ... }
    O->>R: structured tool result
    R-->>O: "I have 7:15 and 7:45 open — which works?"
    O-->>API: reply + trace
    API-->>FE: 0:"text"  ·  8:[trace]  ·  d:stop
    FE-->>U: streamed reply + tool-trace panel
```

The Planner **never speaks to the user** and the Responder **never decides
actions** — this keeps prompts small, output easy to validate, and failures
isolated. The stream uses the [Vercel AI SDK data-stream protocol](docs/sse_protocol.md)
so the UI can render a live **tool-trace panel** from the same response.

### Three-phase conversation flow

Memory carries a `phase` that selects which planner prompt is active, so the model
only ever reasons about one step at a time.

```mermaid
stateDiagram-v2
    [*] --> Discovery
    Discovery --> Availability : a specific restaurant is named
    Availability --> Booking : a time slot is chosen
    Booking --> [*] : reservation confirmed

    note right of Discovery
      search_restaurants_by_filters
      recommend_venues
    end note
    note right of Availability
      check_availability
      (date · party size)
    end note
    note right of Booking
      collect email → create_reservation
    end note
```

Python-level guards sit around the LLM: field cleaners drop hallucinated values
(e.g. `party_size = 0`), and interceptors block a `create_reservation` call when
required fields are missing — so a stray model output can't create a bad booking.

### Hybrid semantic retrieval

Search runs as **SQL hard filters + embedding soft-rank**: Postgres narrows by
cuisine/zone/price, then `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
cosine similarity over an in-memory index) ranks the candidates — so
*"romantic dinner"* surfaces the `date-night` venues with zero keyword overlap.
If the embedding model is unavailable (missing dep, disabled via config), the
system degrades gracefully to `ilike` keyword search — the fallback path is
unit-tested and must always pass.

---

## Model selection — the 4 GB question

The hardware is a laptop with a **4 GB VRAM** GPU. That constraint decided the
model strategy, and the eval harness measured it honestly
([full comparison](reports/eval_summary.md), [specs](docs/model_specs.md)):

| | `qwen3:1.7b` (2.0B params) | `qwen3:4b` (4.0B params) |
|---|---:|---:|
| Conversations passed | 32/45 (71.1%) | **42/45 (93.3%)** |
| Booking flows (hardest bucket) | **4/9 (44.4%)** | 7/9 (77.8%) |
| Cancel/modify flows | 9/9 | 9/9 |
| Throughput (warm) | **48.9 tok/s** | 22.3 tok/s |
| GPU placement | **100% GPU** | 71% GPU / 29% CPU |
| On-disk size | 1.4 GB | 2.5 GB |

The 1.7B is 2.2× faster and fully GPU-resident — but it **skips the availability
step and fires premature booking attempts** in 8 of 13 failed conversations.
That failure mode is disqualifying for a booking system, and no amount of
Python guard-railing can force a model to choose a step it decided to skip.
**Decision: `qwen3:4b` is the floor.** The quality/cost knee for multi-step
tool-use planning sits between 2.0B and 4.0B parameters.

- **Quantization floor: `Q4_K_M`** — verified via `ollama show`; never lower.
- The 4B's runtime footprint (weights + 8K-context KV cache ≈ 4.2 GB) exceeds
  the card, so Ollama offloads ~29% to CPU — disclosed, measured, documented.
- **Reasoning disabled** (`think:false`): qwen3 still emits a short `</think>`
  preamble, stripped in code before JSON/prose is used.
- **Next (Phase 8):** QLoRA fine-tuning of the 1.7B on synthetic trajectories,
  to test how far fine-tuning moves that floor.

---

## Evaluation

A re-runnable harness (`scripts/evaluate.py` + 45 multi-turn fixtures in
`tests/eval/conversations.yaml`) scores **tool-call accuracy, slot-filling, and
task completion** across five buckets: search, availability, booking, edge
cases, cancel/modify. Both models ran the identical harness — same prompts,
same Python guards, same fixtures, same hardware.

| Bucket | `qwen3:1.7b` | `qwen3:4b` |
|---|---:|---:|
| Search | 7/9 | **9/9** |
| Availability | 6/9 | **8/9** |
| Booking | 4/9 | **7/9** |
| Edge cases | 6/9 | **9/9** |
| Cancel/modify | **9/9** | **9/9** |
| **Overall** | 32/45 (71.1%) | **42/45 (93.3%)** |

Two findings worth an interview:

1. **The 4B started at 33%.** The harness caught a broken time-normalization
   keystone (planner said `"7:30pm"`, slots stored `"19:30"` — never matched) and
   cancel/modify interceptor gaps. Targeted fixes took it 33% → 93.3% — the eval
   paid for itself before the model comparison even ran.
2. **Cancel/modify is 9/9 on both models.** Those flows are the most
   interceptor-guarded; deterministic Python paths are model-independent. The
   quality delta between models lives almost entirely in *planning depth*.

Reports live in `reports/` (per-run markdown + JSON); the summary is
[`reports/eval_summary.md`](reports/eval_summary.md).

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind, shadcn/ui | Vercel AI SDK v4 `useChat` |
| Backend | FastAPI, Uvicorn | Pydantic schemas, CORS, SSE streaming |
| LLM | Ollama · `qwen3:4b` (selected via [eval](#model-selection--the-4-gb-question)) | reasoning disabled for latency |
| Retrieval | sentence-transformers `all-MiniLM-L6-v2` | hybrid: SQL filters + cosine rank, ilike fallback |
| Data | PostgreSQL 16 (SQLAlchemy) | JSONB columns for cuisines, hours, menu, tags |
| Config | Pydantic `BaseSettings` (`app/config.py`) | single source of truth via `.env` |
| Infra | Docker Compose | Postgres + Ollama containers |

---

## Repository structure

```
.
├── app/
│   ├── agent/            # planner, responder, orchestrator, tools, memory
│   ├── api/              # FastAPI app, routes, schemas, utils (slots, dates, email)
│   ├── data/             # SQLAlchemy models + connection, seed data
│   ├── retrieval/        # embedding index + semantic rank (hybrid search)
│   └── main.py           # legacy Streamlit UI (fallback)
├── frontend/             # Next.js 14 chat UI (primary)
├── scripts/              # db reset, seeding, smoke test, evaluate
├── tests/
│   ├── eval/             # 45 eval fixtures + pure scorer functions
│   └── retrieval/        # semantic-rank recall + fallback tests
├── reports/              # eval reports + model-comparison summary
├── docs/                 # model specs, SSE protocol
└── docker-compose.yml    # Postgres + Ollama
```

---

## Getting started

### Prerequisites

- **Docker** (for Postgres + Ollama) · **Python 3.11+** · **Node 20+** (for the UI)
- Any local Ollama on `:11434` works too (Docker or native)

### 1. Start the local services

```bash
docker compose up -d                                   # Postgres :5432, Ollama :11434
docker exec goodfoods-ollama ollama pull qwen3:4b      # first-time model pull (~2.5 GB)
```

### 2. Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                   # defaults work out of the box

# seed the database
python -m scripts.reset_db
python -m scripts.load_restaurants
python -m scripts.add_opening_hours

# run the API (http://localhost:8000, docs at /docs)
uvicorn app.api.main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev                                            # http://localhost:3000
```

> First message is slow while qwen3 loads; subsequent turns stream normally.
> Prefer the legacy UI? `streamlit run app/main.py` (port 8501).

### Quick checks (no UI)

```bash
python -m scripts.smoke_test --model qwen3:4b          # scripted search → availability → booking
python -m scripts.evaluate --model qwen3:4b            # full 45-conversation eval
pytest tests/retrieval                                 # semantic-rank + fallback tests
```

---

## Configuration

All settings live in `app/config.py` (read from `.env`); every value has a safe
local default. See `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen3:4b` | model tag |
| `OLLAMA_THINK` | `false` | qwen3 reasoning toggle |
| `OLLAMA_TIMEOUT` | `180` | per-request timeout (s) |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/goodfoods` | Postgres DSN |
| `SEMANTIC_SEARCH_ENABLED` | `true` | embedding rank on/off (off → ilike fallback) |
| `GOODFOODS_EMAIL` / `..._PASSWORD` | empty | SMTP (blank = email no-ops) |
| `API_CORS_ORIGINS` | `http://localhost:3000,http://localhost:8501` | allowed origins |

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/agent/chat` | one-shot chat turn → `{ reply, tool_output, phase }` |
| `POST` | `/agent/chat/stream` | streaming chat (AI SDK data-stream) |
| `POST` | `/agent/memory/reset` | clear conversation memory |
| `POST` | `/search` | search restaurants by filters |
| `POST` | `/book` | create a reservation (full validation) |
| `POST` | `/availability/` | check available slots |
| `POST` | `/reservations/` | REST booking (async email) |
| `POST` | `/customers/profile` | save/update customer profile |
| `GET` | `/restaurants/` · `/analytics/` | listings and platform analytics |

### Agent tools

`search_restaurants_by_filters` · `recommend_venues` · `check_availability` ·
`create_reservation` · `cancel_reservation` · `modify_reservation` ·
`get_seating_map` · `get_seating_labels` · `get_amenities` · `get_booking_details`

Cancel/modify operate on the customer's most recent `confirmed` reservation
(resolved by email); cancel is a **soft-delete** (`status = "cancelled"`).

---

## Example conversations

**Quick booking**

> **User:** I'd like to book a table for 4 at GoodFoods Bistro tomorrow.
> **Agent:** What time would you prefer for your party of 4?
> **User:** Around 7 PM.
> **Agent:** I have 6:45 PM and 7:15 PM open — which works best?
> **User:** 7:15. My email is john@example.com.
> **Agent:** Confirmed — 4 at GoodFoods Bistro tomorrow, 7:15 PM. A confirmation email is on the way.

**Discovery (semantic)**

> **User:** Somewhere romantic for an anniversary.
> **Agent:** For a date night, GoodFoods Rooftop (skyline views) and GoodFoods
> Garden (quiet garden seating) come up. Want me to check availability at either?

---

## Design decisions

- **Two agents, not one** — decouples action-selection from language so each prompt
  is small and independently testable; the Planner's JSON is schema-validated.
- **Deterministic guards around the LLM** — field cleaners and interceptors live in
  Python, so the model can't persist invalid state or fire a premature booking.
- **Eval before optimism** — the 45-case harness ran from day one; it caught the
  33%-baseline keystone bug and made the model-size decision a measurement
  instead of a guess.
- **Graceful degradation** — embedding search falls back to keyword search when
  the model is unavailable; email falls back to no-op without SMTP creds.
- **Config as a single source of truth** — no hardcoded model/endpoint/DSN; swapping
  models is one env var (that's how both eval models ran).
- **Local-first** — no external LLM API; data stays on the machine, cost is fixed.

---

## Roadmap

- 🚀 **Deployment (Phase 7)** — Vercel UI + containerized API + recorded demo,
  with a documented bring-your-own-local-Ollama path.
- 🎓 **QLoRA fine-tuning (Phase 8)** — fine-tune the 1.7B planner on synthetic
  trajectories (4-bit NF4 + LoRA, Unsloth) to test how far the model-size floor
  can be pushed. Training data generated independently of the eval fixtures.
- 🍽️ Product ideas — menu browsing and pre-ordering, seat-map visualization,
  richer vibe filters.

## Assumptions & limitations

- Requires a reachable **local Ollama** with the configured model pulled.
- **PostgreSQL** is required (JSONB columns); scripts seed 50 sample restaurants.
- Email delivery needs valid SMTP credentials; without them, bookings still succeed
  and email is skipped gracefully.
- Eval numbers are specific to this hardware (RTX 3050 4 GB) and the Q4_K_M
  quantization floor — documented in [`docs/model_specs.md`](docs/model_specs.md).
- The Streamlit UI is a **legacy fallback** and will be retired once the Next.js UI
  is verified end-to-end.
