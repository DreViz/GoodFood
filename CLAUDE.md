# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoodFoods AI Reservation Agent — a natural-language restaurant booking system powered by a two-agent LLM architecture (Planner + Responder) backed by a local Ollama model (`qwen3:4b` by default; `qwen3:8b` for eval). Users chat via a Streamlit UI; the backend is a FastAPI server with a PostgreSQL database (via SQLAlchemy).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Database setup (run in order after DB is created)
python -m scripts.reset_db
python -m scripts.load_restaurants
python -m scripts.add_opening_hours

# Start the backend API (port 8000)
uvicorn app.api.main:app --reload --port 8000

# Start the Next.js frontend (port 3000) — primary UI, requires Node 20+
cd frontend && npm install && npm run dev

# Start the legacy Streamlit frontend (port 8501) — fallback only
streamlit run app/main.py

# Run tests
pytest

# Phase 1 smoke test (end-to-end search -> availability -> booking on the local model)
python -m scripts.smoke_test --model qwen3:4b

# Seed scripts (optional, for sample data)
python -m scripts.generate_customers
python -m scripts.generate_reservations
python -m scripts.load_customer_preferences
```

### Local services (Docker)

Postgres and Ollama run as containers via `docker-compose.yml`:

```bash
docker compose up -d          # start postgres (:5432) + ollama (:11434)
docker exec goodfoods-ollama ollama pull qwen3:4b   # first-time model pull
```

## Architecture

### Two-Agent LLM Pattern

The core architecture separates **decision-making** from **language generation**:

1. **Planner Agent** (`app/agent/planner_agent.py`) — Calls Ollama to produce a strict JSON object: either `{"plan": "reply", "reply": "..."}` or `{"plan": "execute", "action": "<tool>", "args": {...}}`. The planner never talks to the user directly.

2. **Responder Agent** (`app/agent/responder_agent.py`) — Consumes tool output and generates the natural-language reply sent to the user. For `create_reservation`, it bypasses the LLM and uses deterministic formatting (`format_reservation_message`).

3. **Agent Orchestrator** (`app/agent/agent.py`) — Ties planner → tool execution → responder together. Routes planner output, manages per-user context, and updates conversation memory.

### Three-Phase Conversation Flow

The planner operates in phases controlled by `ConversationMemory.state["phase"]`:

- **Phase 1 — Discovery** (`planner_prompt_phase1.txt`): Extract search filters (cuisine, zone, price, tags), run `search_restaurants_by_filters` or `recommend_venues`. Transitions to Phase 2 when a specific restaurant is named.

- **Phase 2 — Availability** (`planner_prompt_phase2.txt`): Collect date, party_size, and optional time for the selected restaurant. Calls `check_availability`. Transitions to Phase 3 when the user picks a time slot.

- **Phase 3 — Booking** (`planner_prompt_phase3.txt`): Collect missing booking fields (email, seating preference). Confirms reservation via `create_reservation`.

Each phase has its own prompt file in `app/agent/` (`planner_prompt_phase1/2/3.txt`). `planner_agent.py` selects the active prompt via `load_prompt_for_phase()` based on `memory.state["phase"]` (`discovery` → phase1, `availability` → phase2, `booking` → phase3). There is no generic/combined prompt file.

### Conversation Memory

`app/agent/conversation_memory.py` — Pure key-value state (`phase`, `restaurant`, `date`, `time`, `party_size`, `customer_email`, `seating_pref`, etc.). Only the planner updates memory (via `update_from_planner`). No NLP parsing — the LLM decides what to store.

`planner_agent.py` has two layers of safety logic that gate the LLM's output before any tool runs:
- **Field cleaners** (`safe_extract_party_size`, `safe_extract_date`, `safe_extract_time`, `strip_memory_if_unsupported`) filter hallucinated or invalid values (e.g. `party_size=0`) before they reach memory.
- **Hardcoded interceptors** in `call_planner_llm` (after `validate_planner_json`) override specific planner decisions: early email capture in booking phase, blocking `create_reservation` when required fields are missing from memory, and intercepting `get_booking_details` to ask for confirmation instead. These guards live in Python, not in the prompt — editing the prompt alone will not change them.

Note: memory is a single module-level `memory` instance in `planner_agent.py` (used by the agent flow). The frontend calls `POST /agent/memory/reset` to reset it. (The old duplicate `conversation_memory` instance and dead `/agent/memory/update` endpoint in `main.py` were removed in Phase 2 — B6.)

### Tool System

- **`app/agent/tool_calls.py`** — Defines `TOOL_SPEC` (metadata) and `TOOL_FUNCTIONS` (implementations). All tools query PostgreSQL via SQLAlchemy `SessionLocal`. The `dispatch_tool(name, args)` function normalizes args, resolves restaurant names to `location_id` via fuzzy matching, and dispatches to the implementation. There is no separate registry class — `agent.py` calls `dispatch_tool` directly.

The planner's `allowed_actions` whitelist in `planner_agent.py` is the source of truth for what the planner *may* emit; `TOOL_SPEC` in `tool_calls.py` is the source of truth for what is *actually implemented*. As of Phase 2 both `cancel_reservation` and `modify_reservation` are implemented and whitelisted (B5).

Available tools (in `TOOL_SPEC`): `search_restaurants_by_filters`, `recommend_venues`, `check_availability`, `create_reservation`, `get_seating_map`, `get_amenities`, `get_booking_details`, `get_seating_labels`, `cancel_reservation`, `modify_reservation`. Cancel/modify operate on the customer's most recent `status="confirmed"` reservation (resolved by `customer_email`); cancel is a soft-delete (`status="cancelled"`).

### API Layer

`app/api/main.py` — FastAPI app. Routes are mounted under prefixes:

| Prefix | File | Purpose |
|---|---|---|
| `/agent` | `routes/agent.py` | `POST /agent/chat` (non-streaming, `ChatResponse`) + `POST /agent/chat/stream` (Vercel AI SDK data-stream — see `docs/sse_protocol.md`) |
| `/restaurants` | `routes/restaurants.py` | Restaurant listings, slots, analytics |
| `/reservations` | `routes/reservations.py` | CRUD reservations with async email |
| `/customers` | `routes/customers.py` | Customer profile save/lookup |
| `/availability` | `routes/availability.py` | Availability check endpoint |
| `/notifications` | `routes/notifications.py` | Email sending |
| `/analytics` | `routes/analytics.py` | Platform-wide analytics |
| _(root)_ | `routes/shortcuts.py` | `POST /search`, `POST /book` — direct tool shortcuts (used by the eval harness) |

Routers declare **no `prefix=`** themselves — `main.py` supplies it at mount time (Phase 2 B2 fix; declaring both caused `/reservations/reservations/`). Request/response bodies are validated by Pydantic schemas in `app/api/schemas.py` (`ChatRequest`, `ChatResponse`, `SearchRequest`, `AvailabilityRequest`, `BookingRequest`). CORS is configured from `settings.api_cors_origins`.

`POST /agent/memory/reset` is attached directly to the app in `main.py` (not via a router), called by the frontend on page refresh.

`routes/preferences.py` exists on disk but is **not mounted** in `main.py`.

### Database

- **ORM**: SQLAlchemy with `declarative_base` in `app/data/db_connection.py`
- **Models** (`app/data/db_models.py`): `Restaurant` (with JSONB columns for cuisines, amenities, seating_sections, menu, policies, opening_hours, tags), `Customer`, `Reservation`, `CustomerPreferences`
- **Connection**: Defaults to `postgresql://postgres:postgres@localhost:5432/goodfoods` (overridable via `DATABASE_URL` env var)
- **Slot management** (`app/api/utils/slot_manager.py`): Parses `opening_hours` JSON to generate 30-min slots, checks capacity against existing reservations

### Frontend

**Primary UI — `frontend/`** (Phase 5): Next.js 14 (App Router) + Tailwind + shadcn/ui + Vercel AI SDK. `components/chat-panel.tsx` uses `useChat` against `POST /agent/chat/stream` (AI SDK data-stream protocol); `components/tool-trace.tsx` renders the planner/tool trace from the message annotation; `components/preferences-sidebar.tsx` saves via `POST /customers/profile`. Config via `frontend/.env.local` (`NEXT_PUBLIC_API_URL`). Requires Node 20+. See `frontend/README.md`.

**Legacy fallback — `app/main.py`** (Streamlit): still runnable; since Phase 2 it calls the non-streaming `POST /agent/chat`. Slated for removal once the Next.js UI is verified live (Phase 5 task 11, deferred).

## Key Configuration

All runtime configuration lives in **`app/config.py`** (Pydantic `BaseSettings`,
accessed via `get_settings()`) — the single source of truth. Values are read
from environment variables / a `.env` file at the project root, and every field
has a safe local-dev default. See `.env.example` for the full list. **Do not
hardcode config in agent code.**

| Env var | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server base URL (endpoint paths derived in code) |
| `OLLAMA_MODEL` | `qwen3:4b` | Model tag (`qwen3:4b` dev, `qwen3:8b` eval) |
| `OLLAMA_THINK` | `false` | qwen3 reasoning toggle (off = faster/cleaner on CPU) |
| `OLLAMA_TIMEOUT` | `180` | Per-request timeout, seconds (qwen3:4b ≈ 6 tok/s CPU) |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/goodfoods` | PostgreSQL (JSONB columns required) |
| `GOODFOODS_EMAIL` / `GOODFOODS_EMAIL_PASSWORD` | empty | SMTP via Gmail (blank = email no-ops) |
| `API_CORS_ORIGINS` | `http://localhost:3000,http://localhost:8501` | Allowed CORS origins (used from Phase 2) |

- **qwen3 reasoning**: `qwen3` emits a `</think>`-terminated preamble even with
  `think` disabled; `strip_model_reasoning()` (`app/agent/llm_utils.py`) removes
  it before the planner parses JSON and the responder returns prose.
- **Planner fallback**: `app/agent/mock_llm.py` returns a generic greeting when
  Ollama is unreachable.
- **Model specs / quantization floor**: see `docs/model_specs.md`.

## Data Files

- `app/data/goodfoods_locations_unique_50.json` — Restaurant seed data (50 locations)
- `app/data/goodfoods_locations (1).json` — Original full dataset

## Testing

Tests in `tests/` are currently placeholder files (empty). The project uses `pytest` and `httpx` (declared in requirements.txt).
