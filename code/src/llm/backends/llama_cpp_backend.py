"""In-process GGUF backend via llama-cpp-python (the default).

Reuses v1's streaming + TTFT measurement, but fixes the v1 output-token bug: v1 set
``output_tokens = len(full_response)`` (character count). Here we prefer the model's own
``usage`` block when llama.cpp emits one, and otherwise fall back to tokenizing the
generated text — never a character count.

For GPU acceleration, reinstall the wheel with the OS build flag (see README):
  macOS:  CMAKE_ARGS="-DGGML_METAL=on" uv pip install --force-reinstall ...
  Linux:  CMAKE_ARGS="-DGGML_CUDA=on"  uv pip install --force-reinstall ...
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from config.settings import BackendConfig
from llm.streaming import StreamEvent, consume_stream
from llm.types import BackendInfo, LLMRequest, LLMResponse

if TYPE_CHECKING:
    from llama_cpp import Llama


@dataclass(slots=True)
class LlamaCppBackend:
    """Wraps a loaded :class:`llama_cpp.Llama` instance. Satisfies ``LLMBackend``."""

    llm: Llama
    model_id: str
    n_ctx: int

    @classmethod
    def create(cls, cfg: BackendConfig) -> LlamaCppBackend:
        from llama_cpp import Llama

        model_dir = cfg.model_dir or _default_model_dir()
        model_path = (model_dir / cfg.model_id).as_posix()
        if not Path(model_path).exists():
            raise FileNotFoundError(f"GGUF model not found at {model_path}")
        try:
            llm = Llama(
                model_path=model_path,
                n_ctx=cfg.n_ctx,
                n_gpu_layers=cfg.n_gpu_layers,
                n_threads=cfg.n_threads,
                seed=cfg.seed,
                verbose=False,
            )
        except Exception as exc:
            # Most commonly a memory/allocation failure when offloading a large KV cache
            # (n_ctx) to the GPU. Surface an actionable hint instead of a cryptic crash.
            raise RuntimeError(
                f"Failed to load GGUF model at {model_path} "
                f"(n_ctx={cfg.n_ctx}, n_gpu_layers={cfg.n_gpu_layers}): {exc}. "
                f"If this is out-of-memory, lower N_CTX or N_GPU_LAYERS in config/config.py."
            ) from exc
        return cls(llm=llm, model_id=cfg.model_id, n_ctx=cfg.n_ctx)

    def info(self) -> BackendInfo:
        return BackendInfo(name="llama_cpp", model_id=self.model_id)

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def generate(self, request: LLMRequest) -> LLMResponse:
        self._hard_reset()
        # Filled by _events() at the first streamed chunk; read by the input-token fallback.
        prompt_tokens: dict[str, int] = {}
        start = time.perf_counter()
        return consume_stream(
            self._events(request, prompt_tokens),
            start=start,
            fallback_input_tokens=lambda: prompt_tokens.get(
                "evaluated", self._count_request_tokens(request)
            ),
            fallback_output_tokens=self.count_tokens,
        )

    def _events(self, request: LLMRequest, prompt_tokens: dict[str, int]) -> Iterator[StreamEvent]:
        """Translate llama.cpp streamed chunks into normalized :class:`StreamEvent`s."""
        stream = self.llm.create_chat_completion(
            messages=cast(Any, request.messages),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop,
            seed=request.seed,
            stream=True,
        )
        for chunk in cast(Any, stream):
            if "evaluated" not in prompt_tokens:
                # At the first streamed chunk (the role-only chunk), exactly the formatted
                # prompt (chat template + BOS/role tokens included) has been evaluated, so
                # the model's token counter IS the true input size — unlike re-tokenizing
                # message contents, which misses template tokens.
                n_tokens = getattr(self.llm, "n_tokens", None)
                if n_tokens is not None:
                    prompt_tokens["evaluated"] = int(n_tokens)
            choices = chunk.get("choices") or []
            choice = choices[0] if choices else {}
            usage = chunk.get("usage") or {}
            yield StreamEvent(
                content=(choice.get("delta") or {}).get("content"),
                finish_reason=choice.get("finish_reason"),
                usage_output_tokens=(
                    int(usage["completion_tokens"]) if usage.get("completion_tokens") else None
                ),
            )

    def close(self) -> None:
        close = getattr(self.llm, "close", None)
        if callable(close):
            close()

    def _hard_reset(self) -> None:
        """Wipe all model state so each question starts from an empty context.

        Critical for metric integrity: ``Llama.generate`` reuses KV-cache cells for any prompt
        prefix shared with the previous call (prefix-match optimization). Across benchmark
        questions — which share the same system prompt / serialized data — that would make
        TTFT and prompt-eval counts depend on question order and let state bleed between
        questions. We therefore physically clear the KV cache and zero the token counter
        before every generation, so no context leaks from one question to the next.
        """
        # Physically clear the KV cache (reset() alone only does this for recurrent models).
        ctx = getattr(self.llm, "_ctx", None)
        kv_clear = getattr(ctx, "kv_cache_clear", None)
        if callable(kv_clear):
            kv_clear()
        self.llm.reset()  # n_tokens = 0 (also disables the prefix-match path)

    def _count_request_tokens(self, request: LLMRequest) -> int:
        joined = "\n".join(message["content"] for message in request.messages)
        return self.count_tokens(joined)


def _default_model_dir() -> Path:
    from config.paths import ProjectPaths

    return ProjectPaths.discover().models_dir
