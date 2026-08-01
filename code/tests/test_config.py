"""Tests for config wiring: config.py constants flow into the typed dataclasses."""

from __future__ import annotations

import pytest

from config.config import (
    BACKEND,
    MAX_OUTPUT_TOKENS,
    N_GPU_LAYERS,
    SEED,
    TEMPERATURE,
)
from config.settings import BackendConfig, _validated_backend_kind


def test_backend_config_defaults_from_config() -> None:
    cfg = BackendConfig(model_id="test.gguf")
    assert cfg.kind == BACKEND
    assert cfg.n_gpu_layers == N_GPU_LAYERS
    assert cfg.temperature == TEMPERATURE
    assert cfg.seed == SEED
    # MAX_OUTPUT_TOKENS = 0 means uncapped and is normalized to None; positive passes through.
    expected = MAX_OUTPUT_TOKENS if MAX_OUTPUT_TOKENS > 0 else None
    assert cfg.max_tokens == expected


def test_backend_config_override() -> None:
    cfg = BackendConfig(model_id="q.gguf", n_gpu_layers=0, max_tokens=512)
    assert cfg.n_gpu_layers == 0  # explicit CPU override
    assert cfg.max_tokens == 512


def test_backend_kind_validation_rejects_typo() -> None:
    with pytest.raises(ValueError, match="Invalid BACKEND"):
        _validated_backend_kind("lama_cpp")


def test_backend_kind_validation_accepts_known_kinds() -> None:
    assert _validated_backend_kind("llama_cpp") == "llama_cpp"
    assert _validated_backend_kind("openai_compat") == "openai_compat"
