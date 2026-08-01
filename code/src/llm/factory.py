"""Construct the configured LLM backend.

Imports are lazy (inside each branch) so that installing only one backend's dependencies
— e.g. ``llama-cpp-python`` but not a running server — is enough to use that backend.
"""

from __future__ import annotations

from config.settings import BackendConfig
from llm.backend import LLMBackend


def get_backend(cfg: BackendConfig) -> LLMBackend:
    """Build the backend selected by ``cfg.kind``."""
    match cfg.kind:
        case "llama_cpp":
            from llm.backends.llama_cpp_backend import LlamaCppBackend

            return LlamaCppBackend.create(cfg)
        case "openai_compat":
            from llm.backends.openai_compat_backend import OpenAICompatBackend

            return OpenAICompatBackend.create(cfg)
        case _:  # pragma: no cover - guarded by the BackendKind Literal
            raise ValueError(f"Unknown backend kind: {cfg.kind!r}")
