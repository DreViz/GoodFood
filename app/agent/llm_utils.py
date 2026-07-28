# app/agent/llm_utils.py
"""
Small shared helpers for talking to the Ollama LLM.

qwen3 is a reasoning model: even with "think" disabled it emits a short
reasoning preamble terminated by a lone ``</think>`` tag before the real
answer. Both the planner (which needs clean JSON) and the responder (which
needs clean prose) must strip that preamble before using the output.
"""
import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_model_reasoning(text: str) -> str:
    """
    Remove qwen3-style reasoning from a model response.

    Handles both a bare preamble ending in ``</think>`` (Ollama with
    think=false) and a well-formed ``<think>...</think>`` block.
    """
    if not text:
        return text

    # Keep only what follows the final closing tag (covers the think=false case
    # where there is no opening <think> tag).
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    text = _THINK_BLOCK.sub("", text)

    return text.strip()
