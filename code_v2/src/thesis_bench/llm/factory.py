"""Build an ``LLMClient`` from a ``ModelConfig``.

The factory is the only place that maps a ``Backend`` enum to a concrete client.
Only Ollama is implemented today; the other backends raise a clear
``NotImplementedError`` so adding one later is a localized change (implement the
``LLMClient`` Protocol, then register it here) — no pipeline code is touched.
"""

from __future__ import annotations

from ..config import ModelConfig
from ..enums import Backend
from .base import LLMClient
from .ollama_client import OllamaClient


def build_client(model: ModelConfig) -> LLMClient:
    match model.backend:
        case Backend.OLLAMA:
            return OllamaClient(model_id=model.model_id, host=model.host)
        case Backend.LLAMA_CPP:
            raise NotImplementedError(
                "llama-cpp-python backend is not implemented yet. "
                "Implement an LLMClient wrapping llama_cpp.Llama and register it here."
            )
        case Backend.LLAMA_SERVER:
            raise NotImplementedError(
                "llama.cpp server backend is not implemented yet. "
                "Implement an LLMClient against the OpenAI-compatible endpoint and "
                "register it here."
            )
