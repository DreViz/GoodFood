# GoodFoods AI Reservation Agent

An AI-powered restaurant reservation system that provides a natural-language interface for booking tables, checking availability, and getting personalized recommendations.


## Demo Video is attached in the email


## Features

- Natural-language booking conversations
- Personalized restaurant recommendations
- Real-time availability checks
- Automated email notifications
- Customer preference tracking

## Setup Instructions

1. Clone the repository

```powershell
git clone https://github.com/DreViz/goodfoods-ai-reservation-agent.git
cd goodfoods-ai-reservation-agent
```

2. Create and activate a Python virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```powershell
pip install -r requirements.txt
```

4. Database setup (local development)

```powershell
python scripts/reset_db.py
python scripts/load_restaurants.py
python scripts/add_opening_hours.py
```

5. Start the application

```powershell
python app/main.py
```

## Prompt engineering and agent architecture

The system uses a two-agent approach to separate decision-making from user-facing language:

- Planner agent: decides the next system action and returns a strict JSON object describing either a follow-up question or a tool call.
- Responder agent: consumes tool output and user context and generates natural, concise replies that continue the conversation.

Key principles:

- Clear separation of responsibilities (planning vs. responding).
- Structured internal communication (JSON) to minimize parsing errors.
- Human-first responses: do not expose internals or mention "tools".
- Progressive information gathering: ask one specific missing question at a time.

## Example conversations

### Journey 1 — Quick booking

User: Hi, I'd like to book a table for 4 at GoodFoods Bistro tomorrow.

Agent: What time would you prefer for your party of 4 tomorrow?

User: Around 7 PM.

Agent: I found availability at 6:45 PM and 7:15 PM. Which time works best for you?

User: 7:15 works.

Agent: Please provide your email address to confirm the reservation.

User: john.doe@example.com

Agent: Your reservation for 4 at GoodFoods Bistro tomorrow at 7:15 PM is confirmed. A confirmation email has been sent to john.doe@example.com.

### Journey 2 — Discovery and details

User: I'm looking for a nice Italian place in Koramangala.

Agent: Here are a few Italian places in Koramangala:
- GoodFoods Bistro — cozy, handmade pasta (mid-range)
- GoodFoods Trattoria — family-style dining, great pizzas (higher-end)

User: Tell me more about GoodFoods Bistro.

Agent: GoodFoods Bistro is known for handmade pasta and wood-fired pizza. They have outdoor seating and a relaxed atmosphere. Would you like me to check availability?

## Error handling

- Planner fallback: a mock planner output is available when the planner fails.
- Tool calls are wrapped in try/except blocks and return user-friendly error messages on failure.
- Responder returns a safe fallback message if the language model fails to generate a reply.

## Business strategy

Target market:

- Urban professionals who want convenient reservations.
- Restaurants that want better table utilization and customer insights.

Value proposition:

- For customers: a fast, natural booking flow and personalized recommendations.
- For restaurants: improved table utilization, fewer no-shows, and data-driven insights.

Revenue models:

- Restaurant partnerships and referral commissions.
- Premium user features and concierge subscriptions.
- Data and analytics services for partners.

Growth roadmap:

1. Core experience: perfect booking flow, sign up restaurant partners, gather feedback.
2. Enhancements: personalization, loyalty features, analytics dashboards.
3. Expansion: geographic rollout, integrations, advanced AI features.

## Future improvements

- Vibe-based suggestions and richer filters.
- Menu browsing and pre-ordering support.
- Persistent sidebar filters and preference memory.
- Seat map visualization and live occupancy indicators.
- Table turnover time predictions.

## Assumptions and limitations

- A local Ollama LLM (Llama 3.2) may be assumed in some setups; adjust prompts for your chosen provider.
- A PostgreSQL database is assumed for production; local development uses the project's scripts to seed sample data.
- Email delivery requires valid SMTP configuration.
- Frontend interactive components (sidebar) are under development and may be incomplete.

