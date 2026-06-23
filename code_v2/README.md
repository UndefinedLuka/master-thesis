# thesis_bench

A typed Python benchmark harness comparing two information-retrieval workflows
for querying structured (relational) data with **local, open-weight LLMs**, as
defined in Chapter 3 of the thesis:

1. **Direct Context Inference** — the (sampled) database is serialized into a
   text format, embedded in the prompt, and the model answers a natural-language
   question directly.
2. **Text-to-SQL** — the model receives only the SQL schema, generates a query,
   which is executed against the real database. A SQL execution error is fed back
   to the model for up to **3 retries** (error-correction loop) before the case
   is marked a definitive failure.

Both pipelines are scored against the same executed ground truth.

## Why Ollama for the LLM backend

The LLM backend sits behind a small `LLMClient` Protocol (`llm/base.py`), so it
is swappable. The default is **Ollama**; the table below records the trade-offs
that drove that choice.

| Backend | Pros | Cons |
|---|---|---|
| **Ollama (default)** | One-line install on Linux & macOS; auto-uses Metal (macOS) / CUDA (Linux); manages GGUF model tags; exposes native `prompt_eval_count`/`eval_count` (true input/output tokens) and `*_duration` (ns) timings for TTFT and throughput | Runs as a separate daemon; quantization is pinned via the model tag rather than a raw file |
| llama-cpp-python | In-process; same tokenizer for exact counts; pinned GGUF files = maximum reproducibility | Native build is finicky per platform (Metal vs CUDA vs CPU BLAS); you manage model lifecycle/memory |
| llama.cpp server | Decoupled; precise llama.cpp flags; detailed timings | You build/manage the server process and ports |
| Transformers (HF) | Largest model coverage; fine-grained control | bitsandbytes is GPU-centric; macOS MPS/CPU support is patchy and slow — weakest fit for "local on both Linux & macOS" |

To swap backends later, implement `LLMClient` and register it in `llm/factory.py`;
no pipeline code changes.

## Architecture

```
src/thesis_bench/
  enums.py          # SerializationFormat, PipelineKind, DatasetKind, Backend, ...
  config.py         # frozen dataclasses: GenerationOptions, ModelConfig, RunConfig, BenchmarkConfig
  errors.py         # typed exception hierarchy
  records.py        # BenchmarkRecord (one NDJSON row) + JSON serialization
  paths.py
  datasets/         # Spider + BIRD loading, ground-truth construction (deterministic sampling)
  serialization/    # snapshot + 6 in-memory serializers (json/xml/yaml/csv/markdown/toon) + registry
  db/               # sqlite query, schema dump, deterministic sampling, materialize_sample
  llm/              # LLMClient Protocol, Ollama client, backend factory
  prompts/          # typed template loader + direct/sql/sql_retry templates
  pipelines/        # Pipeline ABC + direct + text_to_sql (retry loop)
  eval/             # ROUGE-L (core), execution accuracy, BERTScore (optional post-pass)
  runner/           # matrix orchestrator + resumable NDJSON storage
  cli.py            # `thesis-bench run | list-models | score-bertscore`
```

### Design principles

- **Typed so wrong parameters can't be passed.** Closed sets are `Enum`s, configs
  and results are `frozen` dataclasses with `tuple` collections, and the backend /
  serializer boundaries are `Protocol`s. The codebase type-checks under
  `pyright --strict`.
- **One in-memory snapshot per database.** All six serializers and the SQL
  executor consume the *same* `DatabaseSnapshot`, so the BIRD sample is identical
  across formats and across both pipelines (ch-03 requirement).
- **Native metrics.** Token counts come from the backend's own counters and TTFT
  excludes one-time model-load latency.

## Setup

Prerequisites: [`uv`](https://docs.astral.sh/uv/) and, for live runs,
[Ollama](https://ollama.com/) running locally.

```bash
uv sync                      # install core dependencies
ollama pull qwen2.5:7b-instruct-q8_0   # or any open-weight model tag

# Optional: BERTScore (pulls torch + transformers)
uv sync --extra bertscore
```

Place datasets under `data/` (overridable in `DatasetConfig`):

```
data/spider/  data/bird/
```

## Running

```bash
uv run thesis-bench list-models                    # preflight: models available in Ollama
uv run thesis-bench run --config <your_config>     # run the experiment matrix
uv run thesis-bench score-bertscore results/*.ndjson   # optional post-hoc BERTScore
```

Results are written as NDJSON, one file per matrix cell
(`<dataset>__<format>__<model>__<pipeline>.ndjson`), and runs are resumable —
already-completed questions are skipped.

## Development

```bash
uv run pyright src       # strict type check
uv run ruff check src    # lint
uv run pytest            # unit tests (use a fake LLMClient, no daemon needed)
```
