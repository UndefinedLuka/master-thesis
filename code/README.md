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
  metrics/        normalized EM / rouge / bertscore / execution + scoring orchestration
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
uv sync                        # core deps + dev (ruff, pytest)
uv sync --extra metrics        # + torch, transformers, rouge-score, bert-score (scoring only)
uv sync --extra visualizations # + pandas, seaborn, matplotlib (analysis only)
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

# 5b. Report serialized database-payload tokens for all tokenizer families
uv run python -m cli.report_serialization_tokens

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
uv run python -m cli.score_results --results results/ --no-bertscore   # normalized EM + ROUGE-L + exec
uv run python -m cli.score_results --results results/spider/direct/json/model.ndjson \
       --references data/spider_references.ndjson                      # explicit references

```

The `visualizations` package exposes one orchestrator command for all results-chapter artifacts:

```bash
uv run --extra visualizations thesis-visualize-all
```

The command loads the scored benchmark tree and detailed serialization-token CSV once, validates
both complete experiment matrices, and invokes the independent table/figure generators. It writes
`analysis/tables/experiment_outcomes.tex`,
`analysis/tables/serialization_efficiency.tex`,
`analysis/figures/tokenizer_family_comparison.pdf`,
`analysis/figures/direct_accuracy_by_format.pdf`,
`analysis/figures/direct_accuracy_by_model_size.pdf`,
`analysis/figures/direct_accuracy_by_database_size_nem.pdf`,
`analysis/figures/direct_accuracy_by_database_size_bertscore.pdf`,
`analysis/figures/direct_accuracy_by_database_size_rouge_l.pdf`,
`analysis/tables/direct_operational_efficiency.tex`,
`analysis/figures/direct_token_usage_by_format.pdf`,
`analysis/figures/direct_latency_by_format.pdf`,
`analysis/tables/text_to_sql_retrieval_accuracy.tex`,
`analysis/figures/text_to_sql_accuracy_by_model_size.pdf`,
`analysis/tables/text_to_sql_operational_efficiency.tex`,
`analysis/figures/text_to_sql_token_usage.pdf`,
`analysis/figures/text_to_sql_latency.pdf`,
`analysis/tables/text_to_sql_error_correction.tex`, and
`analysis/figures/text_to_sql_error_correction.pdf`. The direct-context accuracy figures report
normalized exact match, BERTScore F1, and ROUGE-L F1. The format figure averages over retained
models and questions, while the model-size figure averages each model over all six formats and
questions. The three database-size figures separately report NEM, BERTScore, and ROUGE-L.
Each contains six format panels, and every panel places all Spider and BIRD databases on the same
logarithmic axis using that format's mean tokenizer-specific token footprint as effective size.
Explicit zero scores assigned to failed runs remain included in the accuracy figures. The operational
efficiency artifacts instead use a common complete-case cohort within each dataset, ensuring
that every format is averaged over the same successfully processed questions and models. The
latency figure decomposes total response latency into time to first token and generation after
the first token. Text-to-SQL operational artifacts include all correction attempts: tokens and
model time are accumulated, and the retry-aware latency figure also includes SQL execution. The
error-correction artifacts report the terminal execution attempt and exact model-generation
time added by attempts after the first.

Qwen 3.5 27B is excluded from the outcome and retrieval-accuracy analyses by default; use
`--exclude-model` to override this. `--context-window` changes the serialization-fit threshold,
while `--tables-dir` and `--figures-dir` change artifact locations. Use `--allow-incomplete` only
for smoke tests with partial inputs.

Internally, each artifact has its own module and produces exactly one table or figure. Shared
publication labels, orderings, colors, filenames, and experiment constants are defined in
`visualizations/config.py`; only `visualizations/generate_all.py` owns a command-line interface.

### Serialization input-token overhead report

`cli.report_serialization_tokens` independently loads every unique database referenced by the
Spider and BIRD dev splits, applies the same six serializers as the direct benchmark, and counts
each serialized payload with one representative GGUF tokenizer from every experiment
family: Qwen 3.5, Ministral 3, and Gemma 4. Models of different sizes in one family share that
family's tokenizer, so they do not need to be loaded separately. Spider databases are serialized
in full; BIRD uses the same materialized sampled databases as the benchmark. GGUFs are opened
with `vocab_only=True`; no inference weights or GPU are needed, and prompts, questions, and
generated output are not counted.

```bash
# Full report: both datasets x all formats x all three tokenizer families
uv run python -m cli.report_serialization_tokens
# equivalent installed console command
uv run thesis-report-serialization-tokens

# Fast end-to-end check, not suitable for reporting
uv run python -m cli.report_serialization_tokens --limit 1

# Runs resume from the detailed CSV checkpoint; force a rebuild after changing serializers/data
uv run python -m cli.report_serialization_tokens --no-resume

# Narrow a run or provide an explicit representative GGUF
uv run python -m cli.report_serialization_tokens --dataset spider --family qwen-3.5 \
  --tokenizer-model qwen-3.5=models/Qwen3.5-0.8B-Q4_K_M.gguf
```

A full unfiltered run writes `results/input_token_overhead/serialization_tokens_by_database.csv`.
It contains one row per database, format, and tokenizer family, including character and UTF-8 byte
sizes alongside the exact token count. The CSV is atomically checkpointed after every tokenizer
count so interrupted runs resume safely.

Limited or filtered runs use a deterministic child of `results/input_token_overhead/subsets/` by
default, so a smoke test cannot overwrite the complete report. Pass `--output-dir` to choose
another location.

Console scripts (`thesis-download`, `thesis-sample-dbs`, `thesis-references`,
`thesis-convert`, `thesis-report-serialization-tokens`, `thesis-benchmark`, `thesis-run-all`,
`thesis-score`, and `thesis-visualize-all`) are equivalent to the `python -m` forms.

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

- **Normalized Exact Match**, **ROUGE-L**, and **BERTScore** on prediction vs reference text —
  the raw answer for Direct, the executed rows (stringified) for Text-to-SQL. For Direct results,
  an empty gold SQL answer is represented as the prompt's canonical no-answer value, `N/A`, so a
  correctly abstaining model is not compared against an empty string. Normalized Exact Match uses
  the established SQuAD normalization: lowercase, remove punctuation and English articles, and
  collapse whitespace before binary exact comparison. BERTScore uses `roberta-large` with
  `en` and baseline rescaling disabled; its precision, recall, and F1 are the raw computed values,
  without clipping. These similarities are not probabilities and raw values are not guaranteed to
  be nonnegative. Like other text-similarity metrics, BERTScore can still score an incorrect answer
  highly when it contains similar wording or entities.
- **Execution success** for Text-to-SQL: binary `1.0` when the generated query executes
  successfully within the retry cap, otherwise `0.0` (stored as `execution_accuracy`). This
  operational measure does not assess answer correctness; both pipelines use the three
  text-based metrics above to assess reference-answer agreement.

When a pipeline, backend, or SQL execution fails and a usable gold reference exists, every enabled
accuracy metric is explicitly set to zero rather than comparing placeholder output. Records without
a usable reference are not scored. With `--no-bertscore`, BERTScore fields are `null` for scored
records; failure zeroes apply to enabled metrics only.

References resolve automatically from `data/<dataset>_references.ndjson` per record, or
pass `--references <file>` explicitly. Already-scored files are skipped unless
`--overwrite`. With `--no-bertscore` the `bertscore_*` fields are written as `null` (never
`0.0`), so "not computed" stays distinguishable from "scored zero".

**Rescoring existing results:** files produced with the previous baseline-rescaled BERTScore
must be updated consistently before comparing or combining them with raw scores. Move or copy
prior scored files outside the `results/` tree first so recursive discovery cannot find duplicate
inputs, then run a full overwrite rescore:

```bash
uv run python -m cli.score_results --results results/ --overwrite
```

This regular `score_results` command supports full rescoring; keep the old files outside `results/`
until the new scores have been validated.

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
