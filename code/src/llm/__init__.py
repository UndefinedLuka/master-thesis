"""The swappable LLM wrapper.

Use :func:`get_backend` to construct a backend from config, then call ``generate`` with an
:class:`LLMRequest`. All concrete backends conform to the :class:`LLMBackend` protocol.
"""

from llm.backend import LLMBackend
from llm.factory import get_backend
from llm.types import (
    BackendInfo,
    ChatMessage,
    LLMMetrics,
    LLMRequest,
    LLMResponse,
)

__all__ = [
    "LLMBackend",
    "get_backend",
    "BackendInfo",
    "ChatMessage",
    "LLMMetrics",
    "LLMRequest",
    "LLMResponse",
]
