# app/api/routes/agent.py
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent.agent import process_user_query
from app.agent.planner_agent import memory
from app.api.schemas import ChatRequest, ChatResponse

router = APIRouter()


def _run_turn(message: str, context: str = ""):
    """Run one agent turn; return (reply, tool_output, phase)."""
    output = process_user_query(message, user_id="anonymous", context=context or "")
    reply = output.get("reply", "Sorry, something went wrong.")
    tool_output = output.get("tool_output")
    phase = memory.state.get("phase") or "discovery"
    return reply, tool_output, phase


@router.post("/chat", response_model=ChatResponse, summary="Non-streaming agent chat")
def chat(req: ChatRequest) -> ChatResponse:
    """
    One-shot chat turn returning the full reply. Used by the Phase 3 eval
    harness and any client that does not want to stream.
    """
    reply, tool_output, phase = _run_turn(req.message, req.context or "")
    return ChatResponse(reply=reply, tool_output=tool_output, phase=phase)


# Vercel AI SDK data-stream protocol encoders
# See docs/sse_protocol.md. Format:
#   0:"<json-string>"   text part
#   8:<json-array>      message annotation (binds to the current assistant message)
#   d:<json-object>     finish message part

def _text_part(text: str) -> str:
    return f"0:{json.dumps(text)}\n"


def _annotation_part(obj) -> str:
    return f"8:{json.dumps([obj])}\n"


def _finish_part(reason: str = "stop") -> str:
    return f"d:{json.dumps({'finishReason': reason})}\n"


@router.post("/chat/stream", summary="Streaming agent chat (Vercel AI SDK data-stream)")
def chat_stream(req: ChatRequest):
    """
    Streams the assistant reply in the Vercel AI SDK data-stream format:
      - text parts (`0:`) carry the reply, whitespace-tokenised word-by-word;
      - a message annotation (`8:`) carries the planner action/args/tool result
        and the post-turn phase, bound to this assistant message — this is what
        the Phase 5 tool-trace panel renders without a second round-trip;
      - a finish part (`d:`) terminates the stream.

    The turn is computed fully before streaming (the responder does not
    token-stream from Ollama), so text chunking is cosmetic — matching the
    prior behaviour.
    """
    reply, tool_output, phase = _run_turn(req.message, req.context or "")

    def stream():
        words = reply.split()
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield _text_part(token)
        yield _annotation_part({
            "type": "agent_trace",
            "phase": phase,
            "tool_output": tool_output,
        })
        yield _finish_part("stop")

    return StreamingResponse(
        stream(),
        media_type="text/plain; charset=utf-8",
        headers={"x-vercel-ai-data-stream": "v1"},
    )
