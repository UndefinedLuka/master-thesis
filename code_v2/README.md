# `code_v2` — Serialization & Text-to-SQL Information Extraction Benchmark

Implements the experiment from Chapter 3 of the thesis: comparing two
information-retrieval pipelines over the **Spider** and **BIRD** datasets, across
**six serialization formats**, using local open-weight LLMs served by **vLLM**.

* **Direct Context Inference** — the serialized database is placed in the model
  context (JSON, XML, YAML, CSV, Markdown, TOON); the model answers in natural
  language.
* **Text-to-SQL** — the model gets the SQL schema, generates a query, which is
  executed against the real SQLite database, with a **3-retry error-correction
  loop**.

Metrics collected per question: input/output tokens (split), Time-To-First-Token,
total generation time (+ query execution time for SQL), execution accuracy and
retry count (SQL), and ROUGE-L + BERTScore against the gold-SQL reference answer.

## Layout

```
src/benchmark/
  config.py     paths.py        sqlite_io.py    serialize.py    datasets.py
  reference.py  llm_client.py   prompts.py      metrics.py      results.py
  viz.py        viz_config.py   prompts_md/{direct_answer,sql_answer}.md
  pipelines/{direct,text_to_sql}.py
  cli/{download,prepare,run,aggregate,visualize}.py
scripts/generate_mock_results.py     # synthetic results for previewing figures
slurm/{env.sh, run_benchmark.sbatch, serve_vllm.sbatch}
tests/{test_serialize,test_metrics,test_text_to_sql,test_viz}.py
config.toml        # experiment settings — the file you edit
viz_config.toml    # figure styling — the other file you edit
```

## Setup

```bash
cd code_v2
uv sync
```

`vllm` is **not** a dependency here — it runs only in the container/server. The
client talks to any OpenAI-compatible endpoint via `--base-url`.

## Workflow

```bash
# 1. Download Spider + BIRD into data/
uv run python -m benchmark.cli.download

# 2. Build reference answers (gold SQL) + serialize every DB to all formats
uv run python -m benchmark.cli.prepare            # add --bird-sample-n 100

# 3. Run a shard against a running vLLM endpoint (see SLURM below)
uv run python -m benchmark.cli.run \
  --model qwen2.5-7b-instruct --dataset spider --pipeline sql \
  --base-url http://localhost:8000/v1
uv run python -m benchmark.cli.run \
  --model qwen2.5-7b-instruct --dataset spider --pipeline direct --format json \
  --base-url http://localhost:8000/v1

# 4. Aggregate all shards -> BERTScore (batched), all_results.ndjson, summary.csv
uv run python -m benchmark.cli.aggregate

# 5. Render the figure suite (vector PDF + PNG) into results/figures/
uv run python -m benchmark.cli.visualize
```

## Visualization

`aggregate` produces the data (`results/all_results.ndjson` + `summary.csv`);
`visualize` turns it into figures. They are decoupled so you can restyle without
recomputing BERTScore.

Everything visual is controlled by **`viz_config.toml`** — edit it and re-run
`visualize`; no code changes needed. It controls:

- **theme**: seaborn `style`/`context`, font, figure size, DPI, despining, bar
  annotations, and the output `formats` (default `["pdf", "png"]` — PDF is vector
  for the thesis).
- **palette**: a vibrant, color-blind-friendly default. Set `palette_name` to any
  named seaborn/matplotlib palette (e.g. `"Set2"`, `"viridis"`, `"flare"`) to swap
  the whole scheme, or list explicit hex colors in `sequence`. `by_format` pins a
  fixed color per serialization format (and `sql`) so each approach is recognizable
  across every figure.
- **figures**: per-figure on/off toggles and the `primary_metric`
  (`bertscore` | `rouge_l`).
- **model_params_b**: parameter counts (billions) per model, used by the
  parameter-scale figure.

Figures generated per dataset: accuracy overview (BERTScore + ROUGE-L), token
consumption (input/output split), latency (TTFT + total time), Text-to-SQL
diagnostics (retry distribution + execution accuracy / result match / failure
rate), accuracy-vs-parameter-scale, quality-vs-output-cost scatter, and a
model×approach accuracy heatmap.

### Preview without real runs

The FRIDA runs do not have to exist yet — generate representative mock results
in the exact result schema, then aggregate + visualize:

```bash
uv run python scripts/generate_mock_results.py     # writes results/*.ndjson
uv run python -m benchmark.cli.aggregate --no-bertscore
uv run python -m benchmark.cli.visualize
```

Runs are **resumable**: each shard streams to `results/<model>_<ds>_<pipeline>[_<fmt>].ndjson`
and re-invoking skips already-completed question indices.

## Running on FRIDA

FRIDA is a SLURM cluster using Enroot + Pyxis containers. Edit `slurm/env.sh`
(model, image, repo path), then:

```bash
sbatch slurm/run_benchmark.sbatch      # single job: serves vLLM + runs the client
tail -f logs/bench-*.out
```

The single-job script starts vLLM in the background, waits for `/health`, then
runs all shards over `localhost`. `slurm/serve_vllm.sbatch` is a long-lived
server alternative (see comments in the file). Changing the model or its
quantization is purely an `env.sh` edit — the Python code never changes.

## Local smoke tests (no GPU)

```bash
uv run pytest                          # serializers + SQL retry loop on a tiny fixture
uv run ruff check src
```

To exercise the client end-to-end without a GPU, point `--base-url` at any
OpenAI-compatible mock or a small local vLLM/llama.cpp server.
