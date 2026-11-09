# Evaluation pipelines

Benchmarks for local, open-weight LLMs on **Spider** and **BIRD**, comparing:

- **Direct Context Inference:** answer questions from databases serialized as JSON, XML,
  YAML, CSV, Markdown, or TOON.
- **Text-to-SQL:** generate SQL from the database schema and execute it with bounded retries.

Spider uses full databases; BIRD uses shared, materialized samples for both pipelines
and reference answers.

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). Run commands from `code/`:

```bash
uv sync
```

Configure the backend, context window, GPU settings, and sampling in
[`src/config/config.py`](src/config/config.py).

- Default backend (`llama_cpp`): place GGUF model files in `models/`.
- OpenAI-compatible server (`openai_compat`): set `BACKEND` and `BASE_URL`.

For FRIDA cluster setup and GPU deployment, see [deploy/README.md](deploy/README.md).

## Run

```bash
# Prepare datasets and reference answers
uv run thesis-download
uv run thesis-sample-dbs
uv run thesis-references

# Run either pipeline (replace model.gguf with your model filename or served name)
uv run thesis-benchmark --dataset spider --pipeline direct --model model.gguf
uv run thesis-benchmark --dataset spider --pipeline text_to_sql --model model.gguf

# Score results: normalized exact match, ROUGE-L, BERTScore, and SQL execution success
uv run --extra metrics thesis-score --results results/
```

Use `--dataset bird` for BIRD, `--limit 10` for a short benchmark, or
`uv run thesis-run-all` to run all models in `models/` across both datasets and pipelines.
Commands accept `--help` for additional options.

Benchmark results are saved to `results/<dataset>/<pipeline>/<format>/<model>.ndjson`
and resume automatically; use `--no-resume` to restart. Scoring writes adjacent
`*.scored.ndjson` files.

## Thesis analysis

Generate tables, figures, and summaries from the complete scored experiment results,
stored reference answers, and serialization-token report. Figure generation requires
LaTeX with `pdflatex`, `pgf`, and `siunitx` v3.

```bash
uv run thesis-report-serialization-tokens
uv run --extra visualizations thesis-visualize-all
```

The token report requires representative GGUF models from each experiment family in
`models/`. Outputs are written to `analysis/tables/`, `analysis/figures/`, and
`analysis/summaries/`; the token report is stored in `results/input_token_overhead/`.

## Development

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
```
