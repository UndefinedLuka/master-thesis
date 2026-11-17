# Analysis of Approaches for Extracting Information from Structured Data Using Large Language Models

Repository for the Master's thesis at the University of Ljubljana, Faculty of Computer and Information Science (2026). It contains the LaTeX sources for the thesis and administrative documents, the Python evaluation framework, raw benchmark results, and the generated tables and figures.

The research benchmarks local open-weight LLMs (Qwen 3.5, Ministral 3, and Gemma 4 families) on relational question answering over **Spider** and **BIRD**, comparing **Direct Context Inference** across six serialization formats (CSV, JSON, Markdown, TOON, XML, YAML) against **Text-to-SQL** with bounded error-correction retries.

## Repository Structure

```text
.
├── code/               # Python evaluation framework (thesis-eval)
│   ├── src/            # Pipelines, LLM backends, serializers, metrics, visualizations, CLI
│   ├── deploy/         # Slurm + Enroot/Pyxis job scripts for the FRIDA cluster
│   ├── results/        # Raw benchmark outputs (*.ndjson)
│   ├── analysis/       # Generated LaTeX tables, PDF figures, and summary CSV/JSON files
│   └── tests/          # Test suite
├── thesis/             # Master's thesis LaTeX sources (thesis_template.tex, chapters/, style/)
├── thesis-proposal/    # Thesis topic proposal LaTeX source and PDF
├── declaration/        # Authorship declaration LaTeX source and PDF
└── LICENSE             # GNU Affero General Public License v3.0
```

- Evaluation codebase and CLI usage: [`code/README.md`](code/README.md)
- FRIDA cluster deployment: [`code/deploy/README.md`](code/deploy/README.md)

## Quick Start

**Compile the thesis** (requires a LaTeX distribution with `latexmk`; generated tables and figures in `code/analysis/` are already included):

```bash
cd thesis
latexmk -pdf thesis_template.tex
```

**Set up the Python environment** (requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)):

```bash
cd code
uv sync
```

See [`code/README.md`](code/README.md) for dataset preparation, benchmarking, scoring, and regenerating thesis figures and tables.

## License

Licensed under the [GNU Affero General Public License v3.0](LICENSE).
