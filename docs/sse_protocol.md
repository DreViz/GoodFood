# Agent streaming wire format (Phase 2 / WS2)

`POST /agent/chat/stream` emits the **Vercel AI SDK data-stream protocol**
(Option X). This lets the Phase 5 Next.js frontend consume the stream with the
AI SDK `useChat` hook directly, and render a tool-trace panel from the same
stream without a second round-trip.

Response header: `x-vercel-ai-data-stream: v1`
Media type: `text/plain; charset=utf-8`

## Parts

Each line is `<code>:<json>\n`:

| Code | Meaning | Payload |
|---|---|---|
| `0:` | text part | JSON string — a chunk of the assistant reply |
| `8:` | message annotation | JSON **array** — bound to the current assistant message |
| `d:` | finish | JSON object — `{"finishReason":"stop"}` |

## What this endpoint emits per turn

1. One `0:"<word> "` text part per whitespace token of the reply (word-by-word;
   the turn is computed fully first, so chunking is cosmetic — the responder
   does not token-stream from Ollama).
2. One `8:[{...}]` message annotation carrying the agent trace (surfaces on the
   assistant message's `annotations` in the AI SDK `useChat` hook):
   ```json
   8:[{"type":"agent_trace","phase":"booking","tool_output":{"action":"create_reservation","args":{...},"result":{...}}}]
   ```
3. `d:{"finishReason":"stop"}` to close the stream.

## Non-streaming alternative

`POST /agent/chat` returns the whole turn in one shot as `ChatResponse`
(`{reply, tool_output, phase}`). The Phase 3 eval harness uses this endpoint.

## Request body

Both endpoints accept `ChatRequest`:
```json
{"message": "Book a table for 2 tomorrow", "session_id": null, "context": ""}
```
