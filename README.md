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

## Features

- 💬 Natural-language booking — search → availability → confirmation in one chat
- 🔎 Filtered discovery and recommendations (cuisine, zone, price, tags)
- 📅 Live availability against real opening hours and seat capacity
- ✏️ Cancel and modify existing reservations (with slot re-validation)
- 📧 Email confirmations (best-effort, non-blocking)
- 🧠 Per-conversation memory with a 3-phase state machine
- 🖥️ Two frontends: a polished Next.js UI and a legacy Streamlit fallback

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
        TOOLS["Tool layer<br/>dispatch_tool"]
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

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind, shadcn/ui | Vercel AI SDK v4 `useChat` |
| Backend | FastAPI, Uvicorn | Pydantic schemas, CORS, SSE streaming |
| LLM | Ollama · `qwen3:4b` (dev) / `qwen3:8b` (eval) | reasoning disabled for latency |
| Data | PostgreSQL 16 (SQLAlchemy) | JSONB columns for cuisines, hours, menu, tags |
| Config | Pydantic `BaseSettings` (`app/config.py`) | single source of truth via `.env` |
| Infra | Docker Compose | Postgres + Ollama containers |

---

## Repository structure

```
.
├── app/
│   ├── agent/            # planner, responder, orchestrator, tools, memory
│   ├── api/              # FastAPI app, routes, schemas, utils (slots, email)
│   ├── data/             # SQLAlchemy models + connection, seed data
│   └── main.py           # legacy Streamlit UI (fallback)
├── frontend/             # Next.js 14 chat UI (primary)
├── scripts/              # db reset, seeding, smoke test
├── docs/                 # model specs, SSE protocol
├── docker-compose.yml    # Postgres + Ollama
└── requirements.txt
```

---

## Getting started

### Prerequisites

- **Docker** (for Postgres + Ollama) · **Python 3.11+** · **Node 20+** (for the UI)

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

> First message is slow on CPU (~30–90 s) while qwen3 loads; subsequent turns are faster.
> Prefer the legacy UI? `streamlit run app/main.py` (port 8501).

### Quick check (no UI, no model)

```bash
python -m scripts.smoke_test --model qwen3:4b          # scripted search → availability → booking
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

## Models & hardware

The system is designed around **running the model yourself**:

- **`qwen3:4b`** — fast dev/iteration model, the default.
- **`qwen3:8b`** — stronger model reserved for evaluation and benchmarking.
- **Quantization floor: `Q4_K_M`** — verified via `ollama show`; never go lower.
- **Reasoning disabled** (`think:false`): qwen3 still emits a short `</think>`
  preamble, which is stripped in code before JSON/prose is used — this keeps
  latency sane and output clean. See [`docs/model_specs.md`](docs/model_specs.md).

Runs on GPU or CPU. CPU inference works but is slow (~6 tok/s for 4B); a single
model at a time is recommended on memory-constrained machines.

---

## Example conversations

**Quick booking**

> **User:** I'd like to book a table for 4 at GoodFoods Bistro tomorrow.
> **Agent:** What time would you prefer for your party of 4?
> **User:** Around 7 PM.
> **Agent:** I have 6:45 PM and 7:15 PM open — which works best?
> **User:** 7:15. My email is john@example.com.
> **Agent:** Confirmed — 4 at GoodFoods Bistro tomorrow, 7:15 PM. A confirmation email is on its way.

**Discovery**

> **User:** Any Italian places in the East zone?
> **Agent:** A couple of options: GoodFoods Bistro (handmade pasta, mid-range) and GoodFoods Trattoria (wood-fired pizza). Want me to check availability at either?

---

## Design decisions

- **Two agents, not one** — decouples action-selection from language so each prompt
  is small and independently testable; the Planner's JSON is schema-validated.
- **Deterministic guards around the LLM** — field cleaners and interceptors live in
  Python, so the model can't persist invalid state or fire a premature booking.
- **Config as a single source of truth** — no hardcoded model/endpoint/DSN; swapping
  models or environments is one env var.
- **Local-first** — no external LLM API; data stays on the machine, cost is fixed.

---

## Business context

**Target users** — urban professionals who want a fast, no-friction way to book;
restaurants that want better table utilization and customer insight.

**Value** — for diners: a natural booking flow with personalized suggestions; for
restaurants: higher utilization, fewer no-shows, and usable data.

**Monetization** — restaurant partnerships and referral commissions, premium
concierge features, and analytics for partners.

---

## Roadmap

- 📊 **Evaluation harness** — 40+ multi-turn conversations scoring tool-call accuracy,
  slot-filling, and task completion (4B vs 8B).
- ⚡ **Quantization benchmark** — latency / throughput / accuracy comparison table.
- 🌐 **Deployment** — Vercel UI + containerized API, with a documented local-Ollama path.
- 🍽️ Menu browsing and pre-ordering, seat-map visualization, richer vibe filters.

## Assumptions & limitations

- Requires a reachable **local Ollama** with the configured model pulled.
- **PostgreSQL** is required (JSONB columns); scripts seed 50 sample restaurants.
- Email delivery needs valid SMTP credentials; without them, bookings still succeed
  and email is skipped gracefully.
- The Streamlit UI is a **legacy fallback** and will be retired once the Next.js UI
  is verified end-to-end.
