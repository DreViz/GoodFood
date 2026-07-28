# GoodFoods — Frontend (Next.js 14)

Recruiter-grade chat UI for the GoodFoods AI reservation agent. Talks to the
FastAPI backend and streams replies via the **Vercel AI SDK** data-stream
protocol (see `../docs/sse_protocol.md`).

## Stack

- Next.js 14 (App Router) · React 18 · TypeScript
- Tailwind CSS 3 + shadcn/ui (Radix)
- Vercel AI SDK v4 (`ai`, `@ai-sdk/react`) — `useChat` against `/agent/chat/stream`

## Prerequisites

- **Node 20+** (this repo uses `nvm`: `nvm use 20`)
- The backend running on `http://localhost:8000` (see repo root README)

## Run

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Configure the backend URL via `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Build

```bash
npm run build        # production build (type-checked + linted)
npm start            # serve the production build
```

## Structure

```
app/
  layout.tsx                 # dark theme, fonts, Toaster
  page.tsx                   # header + chat + preferences layout
components/
  chat-panel.tsx             # useChat client, messages, composer, empty state
  tool-trace.tsx             # collapsible planner/tool trace (AI SDK annotations)
  preferences-sidebar.tsx    # profile form -> POST /customers/profile
  ui/                        # shadcn/ui components
lib/
  api.ts                     # API base URL + reset/profile helpers
```

## Notes

- The reply is computed fully server-side, then streamed word-by-word (the
  responder does not token-stream from Ollama) — chunking is cosmetic.
- The tool-trace panel under each assistant message renders the planner
  `action`/`args`/`result` + phase carried on the message annotation (`8:` part).
