# Evaluation Framework (`thesis-eval`)

Python benchmarking, scoring, and analysis package evaluating local open-weight LLMs on **Spider** (full dev databases) and **BIRD** (materialized sampled dev databases) across two pipelines:

- **Direct Context Inference (`direct`)**: serializes database tables into `csv`, `json`, `markdown`, `toon`, `xml`, or `yaml` and prompts for a direct natural-language answer.
- **Text-to-SQL (`text_to_sql`)**: prompts with the SQLite schema DDL, executes generated SQL against the database, and retries execution errors up to `MAX_RETRIES`.

For Slurm + Enroot/Pyxis execution on the FRIDA cluster, see [`deploy/README.md`](deploy/README.md).

## Directory Structure

```text
code/
├── src/
│   ├── cli/             # Command-line entry points and shared benchmark runner
│   ├── config/          # Experiment constants (config.py), path discovery, dataset registry
│   ├── datasets/        # Spider and BIRD archive downloaders and loaders
│   ├── sqlite/          # SQLite connections, DDL extraction, deterministic row sampling
│   ├── serialization/   # CSV, JSON, Markdown, TOON, XML, YAML serializers + GGUF token counting
│   ├── reference/       # Gold-SQL execution for ground-truth reference answers
│   ├── llm/             # Backend protocol and implementations (llama_cpp, openai_compat)
│   ├── prompts/         # System/user prompt templates (direct_answer.md, sql_answer.md)
│   ├── pipelines/       # Direct Context Inference and Text-to-SQL execution loops
│   ├── metrics/         # Normalized Exact Match, ROUGE-L, BERTScore, SQL execution accuracy
│   ├── results/         # Crash-safe, resumable NDJSON writer and loader
│   └── visualizations/  # LaTeX table, PGF/PDF figure, and summary CSV/JSON generators
├── deploy/              # FRIDA Slurm job scripts and GGUF download manifest
├── analysis/            # Generated tables/, figures/, and summaries/
├── results/             # Benchmark NDJSON outputs (committed) and *.scored.ndjson (gitignored)
├── data/                # Downloaded datasets, sampled BIRD DBs, reference NDJSON (gitignored)
├── models/              # Local GGUF model files (gitignored)
└── tests/               # Offline pytest suite
```

> **Note:** `src/datasets` is a local package on `sys.path` and shadows HuggingFace `datasets`.

## Setup

Requires Python 3.12+ (`.python-version` pins `3.14`) and [`uv`](https://docs.astral.sh/uv/). Run all commands from `code/`:

```bash
uv sync                        # core + dev (pytest, ruff)
uv sync --extra metrics        # + torch, transformers, rouge-score, bert-score
uv sync --extra visualizations # + matplotlib, pandas, statsmodels
```

For GPU inference with the default `llama_cpp` backend:

- **macOS (Apple Silicon / Metal)**: prebuilt wheels include Metal; or rebuild with:
  ```bash
  CMAKE_ARGS="-DGGML_METAL=on" uv pip install --reinstall --no-binary llama-cpp-python llama-cpp-python
  ```
- **Linux (CUDA)**: rebuild from source against the local CUDA toolkit:
  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --reinstall --no-binary llama-cpp-python llama-cpp-python
  ```

## Configuration

Runtime parameters are defined in [`src/config/config.py`](src/config/config.py):

| Constant | Default | Description |
| --- | --- | --- |
| `BACKEND` | `"llama_cpp"` | `"llama_cpp"` (GGUF files in `models/`) or `"openai_compat"` |
| `BASE_URL` / `API_KEY` | `None` | Endpoint URL (e.g. `"http://localhost:8080/v1"`) and key for `openai_compat` |
| `N_CTX` | `131_072` | Context window in tokens |
| `N_GPU_LAYERS` | `-1` | GPU layer offload (`-1` = all layers, `0` = CPU only) |
| `N_THREADS` | `8` | CPU thread count (`llama_cpp` only) |
| `TEMPERATURE` / `SEED` | `0.0` / `0` | Sampling temperature and RNG seed |
| `MAX_OUTPUT_TOKENS` | `2_048` | Per-call output token cap (`0` = uncapped) |
| `BIRD_SAMPLE_ROWS` | `500` | Rows sampled per table when building `data/bird_sampled/` |
| `SAMPLE_SEED` | `0` | RNG seed for BIRD table sampling |
| `MAX_RETRIES` | `3` | Maximum SQL error-correction retries in `text_to_sql` |
| `BERTSCORE_MODEL` | `"roberta-large"` | Model used by `thesis-score` (`BERTSCORE_LANG = "en"`) |

Dataset archive URLs/IDs in [`src/config/registry.py`](src/config/registry.py) can be overridden with the `SPIDER_GDRIVE_ID` and `BIRD_DEV_URL` environment variables.

## Usage

### 1. Prepare Datasets and Reference Answers

```bash
uv run thesis-download     # download and extract Spider + BIRD into data/
uv run thesis-sample-dbs   # materialize sampled BIRD DBs into data/bird_sampled/ (--overwrite to rebuild)
uv run thesis-references   # execute gold SQL -> data/{spider,bird}_references.ndjson
```

Optional pre-serialization of all databases into `data/<dataset>_processed/<db_id>/`:

```bash
uv run thesis-convert
```

### 2. Run Benchmarks

Run a single dataset/pipeline slice:

```bash
uv run thesis-benchmark --dataset spider --pipeline direct --model Qwen3.5-0.8B-Q4_K_M.gguf
uv run thesis-benchmark --dataset bird --pipeline text_to_sql --model Qwen3.5-0.8B-Q4_K_M.gguf
```

- `--formats csv,json,...`: restrict `direct` to a subset of formats (default: `all`).
- `--limit N`: run only the first `N` examples.
- `--no-resume`: overwrite existing slice output instead of resuming.

Run all (or selected) `*.gguf` models in `models/` across both datasets and both pipelines:

```bash
uv run thesis-run-all
uv run thesis-run-all --models Qwen3.5-0.8B-Q4_K_M.gguf,Qwen3.5-2B-Q4_K_M.gguf --limit 10
```

Output layout:

- Direct: `results/<dataset>/direct/<format>/<model-slug>.ndjson`
- Text-to-SQL: `results/<dataset>/text_to_sql/schema/<model-slug>.ndjson`

Runs flush each record and resume automatically by `(example_id, pipeline, format)`.

### 3. Score Results

Computes Normalized Exact Match (`normalized_exact_match`), ROUGE-L F1 (`rouge_l_f`), unrescaled BERTScore (`bertscore_p`, `bertscore_r`, `bertscore_f`), and Text-to-SQL execution success (`execution_accuracy`), writing `<model-slug>.scored.ndjson` alongside each input file:

```bash
uv run --extra metrics thesis-score --results results/
```

- `--references <path>`: score against an explicit reference NDJSON file instead of `data/<dataset>_references.ndjson`.
- `--no-bertscore`: compute Exact Match, ROUGE-L, and execution accuracy only (`bertscore_*` set to `null`).
- `--overwrite`: re-score slices that already have a `.scored.ndjson` file.

### 4. Generate Thesis Tables, Figures, and Audits

Requires `*.scored.ndjson` files under `results/`, reference files in `data/`, LaTeX (`pdflatex`), and either a cached `analysis/summaries/serialization_tokens.csv` or at least one GGUF per model family (`Qwen3.5-*.gguf`, `Ministral-3-*.gguf`, `gemma-4-*.gguf`) in `models/`:

```bash
uv run --extra visualizations thesis-visualize-all
```

Outputs are written to `analysis/tables/*.tex`, `analysis/figures/*.pdf`, and `analysis/summaries/*.{csv,json}`. Key flags: `--refresh-token-cache`, `--allow-incomplete`, `--context-window`.

Additional standalone reporting scripts:

```bash
# Stream per-database serialized token counts as CSV to stdout
uv run thesis-report-serialization-tokens

# Write independent token-count audit to analysis/summaries/serialization_token_audit.json
uv run --extra visualizations python -m cli.audit_serialization_tokens

# Evaluate BIRD row caps (100, 500, 1000) -> analysis/summaries/bird_sampling_sensitivity.json
uv run python -m cli.report_bird_sampling
```

## Development

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
```
