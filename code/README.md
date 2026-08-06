# Evaluation pipelines

Evaluation pipelines for local, open-weight LLMs on structured-data information retrieval
(the methodology of Chapter 3). Two pipelines are compared across six serialization formats
on the **Spider** and **BIRD** datasets:

- **Direct Context Inference** — serialize the whole database into a format, put it in the
  model context, ask the question, get a natural-language answer.
- **Text-to-SQL** — give the model only the schema; it emits SQL, which is executed against
  the real SQLite database, with a bounded error-correction loop (execution errors are fed
  back, up to `MAX_RETRIES` attempts).

BIRD's databases are too large for a context window, so they are sub-sampled **once, to
disk** (`data/bird_sampled/`): serialization, reference answers, and Text-to-SQL query
execution all read the same materialized files, so both pipelines see exactly the same rows.
Spider runs on the full databases.

To run on the FRIDA cluster (Slurm + Enroot/Pyxis), see **[deploy/README.md](deploy/README.md)**.

## Layout

```
src/
  config/         config.py (all experiment constants), paths, typed settings, registry
  datasets/       download (Spider/BIRD) + load + DB-path resolution
  sqlite/         the only package that opens .sqlite (query, schema DDL, sampling, materialize)
  serialization/  the six formats (json, xml, yaml, csv, markdown, toon) + dispatcher
  reference/      gold-SQL execution -> ground-truth rows
  llm/            swappable backend Protocol + factory + backends/{llama_cpp,openai_compat}
  prompts/        direct_answer.md / sql_answer.md + load/render
  pipelines/      direct.py, text_to_sql.py (retry loop), shared result types
  metrics/        rouge / bertscore / execution primitives + scoring orchestration
  results/        resumable NDJSON store
  cli/            one module per command (below) + runner.py (the shared benchmark loops)
deploy/           FRIDA cluster runbook + sbatch templates
```

Dependencies flow one way: `config → datasets/sqlite → serialization → reference/pipelines`,
with `llm` and `prompts` feeding the pipelines, `metrics` pure and I/O-free, and `cli` the
only wiring layer. Adding another inference engine (e.g. vLLM) is one new file in
`llm/backends/` plus one line in `llm/factory.py`.

Note: the local `src/datasets` package shadows HuggingFace's `datasets` on `sys.path` —
don't add a dependency that imports HF `datasets` without renaming the local package first.

## Setup

Requires Python >= 3.12 (local dev pins 3.14 via `.python-version`) and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                 # core deps + dev (ruff, pytest)
uv sync --extra metrics # + torch, transformers, rouge-score, bert-score (scoring only)
```

## Configuration

All experiment constants live in **`src/config/config.py`** — edit that file before a run
instead of passing CLI flags; each constant is documented inline. The ones you will most
likely touch:

- `BACKEND` — `"llama_cpp"` (in-process GGUF, the default) or `"openai_compat"`
  (any server exposing `POST /v1/chat/completions`: llama.cpp server, Ollama, vLLM,
  LM Studio; set `BASE_URL` too).
- `N_CTX` — the fixed context window shared by every model (default 128K); lower it if
  model load runs out of memory.
- `N_GPU_LAYERS` — `-1` = all layers on GPU (Metal/CUDA), `0` = pure CPU.

For `llama_cpp`, put GGUF files under `models/`. The prebuilt wheel ships with Metal on
Apple Silicon; for CUDA on Linux rebuild it:
`CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --force-reinstall --no-binary llama-cpp-python llama-cpp-python`.

**Measurement guarantees.** Every question is generated from a clean context — the
`llama_cpp` backend physically clears the KV cache before each call and `openai_compat`
sends `cache_prompt: false` — so no state or order-dependent TTFT leaks between questions.
Generation is capped at `MAX_OUTPUT_TOKENS` (default 2048; `0` = uncapped). For thinking
models, the `<think>…</think>` phase is timed separately (`reasoning_time_s`), stripped from
the recorded answer, and still counted in `output_tokens`.

## Usage

Dataset URLs / Google Drive ids live in `config/registry.py`, overridable via the
`SPIDER_GDRIVE_ID` / `BIRD_DEV_URL` environment variables — verify them before the first
run; dataset hosts change across releases.

```bash
# 1. Edit src/config/config.py (backend, constants)

# 2. Download all datasets (idempotent; re-running is a no-op)
uv run python -m cli.download_datasets

# 3. Materialize the sub-sampled BIRD databases -> data/bird_sampled/
#    (re-run with --overwrite after changing BIRD_SAMPLE_ROWS / SAMPLE_SEED)
uv run python -m cli.build_sampled_dbs

# 4. Build reference answers -> data/<dataset>_references.ndjson
uv run python -m cli.build_references

# 5. (Optional) materialize all six formats on disk, to eyeball them
uv run python -m cli.convert_data

# 6. Run a pipeline (resumable; results split per slice)
uv run python -m cli.run_benchmark --dataset spider --pipeline direct --model Qwen2.5-7B-Q8_0.gguf
uv run python -m cli.run_benchmark --dataset spider --pipeline direct --model Qwen2.5-7B-Q8_0.gguf --formats json,markdown
uv run python -m cli.run_benchmark --dataset bird  --pipeline text_to_sql --model local
uv run python -m cli.run_benchmark --dataset bird  --pipeline text_to_sql --model local --limit 10  # smoke test

# ... or everything at once (all models in models/ x both datasets x both pipelines)
uv run python -m cli.run_all
uv run python -m cli.run_all --limit 50  # smoke test
uv run python -m cli.run_all --models Qwen3.5-2B-Q4_K_M.gguf,Qwen3.5-4B-Q4_K_M.gguf  # subset

# 7. Score results (needs: uv sync --extra metrics)
uv run python -m cli.score_results --results results/
uv run python -m cli.score_results --results results/ --no-bertscore   # ROUGE-L + exec only
uv run python -m cli.score_results --results results/spider/direct/json/model.ndjson \
       --references data/spider_references.ndjson                      # explicit references
```

Console scripts (`thesis-download`, `thesis-sample-dbs`, `thesis-references`,
`thesis-convert`, `thesis-benchmark`, `thesis-run-all`, `thesis-score`) are equivalent to
the `python -m` forms.

### Results

One NDJSON file per `(dataset, pipeline, format, model)` slice:

```
results/<dataset>/<pipeline>/<format>/<model>.ndjson
  e.g. results/spider/direct/markdown/Qwen2-5-7B-Instruct-Q8_0-gguf.ndjson
       results/bird/text_to_sql/schema/local.ndjson   # Text-to-SQL is schema-only -> "schema"
```

Each record carries the model's full response (`raw_output`), the extracted SQL
(`predicted_sql`, Text-to-SQL only), executed rows, per-attempt details, and the
operational-efficiency metrics (tokens, TTFT, reasoning/generation/query time).

**Resume** is keyed on `(example_id, pipeline, format)` per file — re-running the same
command continues where it stopped; pass `--no-resume` to start fresh. Every record is
flushed to the OS before the run proceeds (a process kill loses nothing) and fsynced at a
fixed interval (bounding hard-crash loss; see `results/store.py`). A truncated final line
from a killed process is isolated and skipped with a warning on the next run. Note:
examples skipped as `context_exceeded` count as completed — delete their lines (or the
slice file) to re-run them after raising `N_CTX`.

### Scoring

`score_results` reads benchmark NDJSON (one file, or a directory tree scored recursively),
joins each record with its reference answer, and writes a `<name>.scored.ndjson` next to
each input:

- **ROUGE-L** + **BERTScore** on prediction vs reference text — the raw answer for Direct,
  the executed rows (stringified) for Text-to-SQL.
- **Execution accuracy** (Spider-style EX) for Text-to-SQL: set-equality of result rows,
  order-sensitive only when the gold SQL has a top-level `ORDER BY`.

References resolve automatically from `data/<dataset>_references.ndjson` per record, or
pass `--references <file>` explicitly. Already-scored files are skipped unless
`--overwrite`. With `--no-bertscore` the `bertscore_*` fields are written as `null` (never
`0.0`), so "not computed" stays distinguishable from "scored zero".

## Tests & lint

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
```

The suite is hermetic — no network, models, or datasets; everything runs against `tmp_path`
fixtures and hand-written fakes (a scripted backend for the retry loop, `httpx.MockTransport`
for the HTTP backend, a tiny on-disk SQLite database). `test_score_results_cli.py` and
`test_metrics_real.py` additionally need the `[metrics]` extra and skip themselves without it.
