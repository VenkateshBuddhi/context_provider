"""Synthesis package interfacing with local generative LLMs (llama3.1 via Ollama)."""
from context_provider.synthesis.gateway import SynthesisGateway
from context_provider.synthesis.prompts import (
    EPISODE_SUMMARY_PROMPT,
    TRAPS_AVOIDED_PROMPT,
    CONTEXT_FOCUS_PROMPT,
)

__all__ = [
    "SynthesisGateway",
    "EPISODE_SUMMARY_PROMPT",
    "TRAPS_AVOIDED_PROMPT",
    "CONTEXT_FOCUS_PROMPT",
]
