"""Experiment constants — the single place to adjust settings before a run.

Change values here rather than passing CLI flags. Every constant in the project
that is not structural (dataset paths, dataclass definitions) lives here.
"""

# ── LLM backend ──────────────────────────────────────────────────────────────
BACKEND: str = "llama_cpp"  # "llama_cpp" | "openai_compat"
BASE_URL: str | None = None  # openai_compat only, e.g. "http://localhost:8080/v1"
API_KEY: str | None = None  # openai_compat only; usually unused for local servers

N_CTX: int = 131_072  # context window (tokens); fixed across all models
N_GPU_LAYERS: int = -1  # -1 = all layers on GPU (Metal/CUDA); 0 = CPU only
N_THREADS: int = 8  # CPU threads (llama_cpp only)
TEMPERATURE: float = 0.0  # 0 = deterministic (greedy)
SEED: int = 0
MAX_OUTPUT_TOKENS: int = 2_048  # per-call generation cap; 0 = uncapped

# ── Dataset ───────────────────────────────────────────────────────────────────
BIRD_SAMPLE_ROWS: int = 500  # rows per table sampled for BIRD context + references
SAMPLE_SEED: int = 0  # random seed for BIRD row sampling

# ── Benchmark ─────────────────────────────────────────────────────────────────
MAX_RETRIES: int = 3  # Text-to-SQL error-correction loop cap

# ── Scoring ───────────────────────────────────────────────────────────────────
BERTSCORE_MODEL: str = "roberta-large"
BERTSCORE_LANG: str = "en"
