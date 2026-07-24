# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoodFoods AI Reservation Agent — a natural-language restaurant booking system powered by a two-agent LLM architecture (Planner + Responder) backed by a local Ollama model (Llama 3.2 3B). Users chat via a Streamlit UI; the backend is a FastAPI server with a PostgreSQL database (via SQLAlchemy).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Database setup (run in order after DB is created)
python -m scripts.reset_db
python -m scripts.load_restaurants
python -m scripts.add_opening_hours

# Start the backend API (port 8000)
python -m app.api.main
# or via uvicorn:
uvicorn app.api.main:app --reload --port 8000

# Start the Streamlit frontend (port 8501)
streamlit run app/main.py

# Run tests
pytest

# Seed scripts (optional, for sample data)
python -m scripts.generate_customers
python -m scripts.generate_reservations
python -m scripts.load_customer_preferences
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

Each phase has its own prompt file in `app/agent/`. The generic `planner_prompt.txt` combines all phases but is **not used at runtime** — `planner_agent.py` selects the phase-specific file via `load_prompt_for_phase()`.

### Conversation Memory

`app/agent/conversation_memory.py` — Pure key-value state (`phase`, `restaurant`, `date`, `time`, `party_size`, `customer_email`, `seating_pref`, etc.). Only the planner updates memory (via `update_from_planner`). No NLP parsing — the LLM decides what to store.

The `planner_agent.py` has safety functions (`safe_extract_party_size`, `safe_extract_date`, `strip_memory_if_unsupported`) that prevent the LLM from storing hallucinated or invalid values into memory.

### Tool System

- **`app/agent/tool_calls.py`** — Defines `TOOL_SPEC` (metadata) and `TOOL_FUNCTIONS` (implementations). All tools query PostgreSQL via SQLAlchemy `SessionLocal`. The `dispatch_tool` function normalizes args and resolves restaurant names to `location_id`.
- **`app/agent/tool_manager.py`** — Generic tool registry (`ToolManager` class). Tools are registered at import time in `agent.py`.

Available tools: `search_restaurants_by_filters`, `recommend_venues`, `check_availability`, `create_reservation`, `get_seating_map`, `get_amenities`, `get_booking_details`, `get_seating_labels`.

### API Layer

`app/api/main.py` — FastAPI app. Routes are mounted under prefixes:

| Prefix | File | Purpose |
|---|---|---|
| `/agent` | `routes/agent.py` | SSE streaming chat endpoint (`/agent/chat/stream`) |
| `/restaurants` | `routes/restaurants.py` | Restaurant listings, slots, analytics |
| `/reservations` | `routes/reservations.py` | CRUD reservations with async email |
| `/customers` | `routes/customers.py` | Customer profile save/lookup |
| `/availability` | `routes/availability.py` | Availability check endpoint |
| `/notifications` | `routes/notifications.py` | Email sending |
| `/analytics` | `routes/analytics.py` | Platform-wide analytics |
| `/preferences` | `routes/preferences.py` | Customer preference storage |

### Database

- **ORM**: SQLAlchemy with `declarative_base` in `app/data/db_connection.py`
- **Models** (`app/data/db_models.py`): `Restaurant` (with JSONB columns for cuisines, amenities, seating_sections, menu, policies, opening_hours, tags), `Customer`, `Reservation`, `CustomerPreferences`
- **Connection**: Defaults to `postgresql://postgres:postgres@localhost:5432/goodfoods` (overridable via `DATABASE_URL` env var)
- **Slot management** (`app/api/utils/slot_manager.py`): Parses `opening_hours` JSON to generate 30-min slots, checks capacity against existing reservations

### Frontend

`app/main.py` — Streamlit app. Communicates with the FastAPI backend via SSE (`/agent/chat/stream`). Sidebar collects user preferences and saves via `/customers/profile`. Resets conversation memory on page refresh via `/agent/memory/reset`.

## Key Configuration

- **LLM**: Ollama at `http://127.0.0.1:11434`, model `llama3.2:3b`
- **Email**: SMTP via Gmail (`GOODFOODS_EMAIL`, `GOODFOODS_EMAIL_PASSWORD` env vars)
- **Database**: `DATABASE_URL` env var (PostgreSQL required for JSONB columns)
- **Planner fallback**: `app/agent/mock_llm.py` returns a generic greeting when Ollama is unreachable

## Data Files

- `app/data/goodfoods_locations_unique_50.json` — Restaurant seed data (50 locations)
- `app/data/goodfoods_locations (1).json` — Original full dataset

## Testing

Tests in `tests/` are currently placeholder files (empty). The project uses `pytest` and `httpx` (declared in requirements.txt).
