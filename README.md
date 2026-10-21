# Master's Thesis Repository

This repository contains the research, codebase, and document sources for the Master's Thesis titled **"Analysis of approaches for extracting information from structured data using large language models"** .

## About the Thesis
The thesis focuses on evaluating and benchmarking how various Large Language Model (LLM) architectures perform when extracting information from structured data sources (such as SQLite databases, JSON, and CSV files). The goal is to analyze the impact of different data serialization methods and model capabilities to determine the most effective approaches for reliable information extraction.

## Required Programs & Setup

To use the code and compile the thesis in this repository, you will need the following programs:

### 1. `uv` (Python Package Manager)
The Python codebase relies on `uv` for environment and dependency management.

**Setup Instructions:**
- Install `uv` by following the instructions on their [official GitHub](https://github.com/astral-sh/uv)
- Navigate to the `code/` directory and install the project dependencies:
  ```bash
  cd code
  uv sync
  ```
- You can then run scripts using `uv run <script_path>`.

### 2. LaTeX
The thesis document is written in LaTeX. You will need a LaTeX distribution (like TeX Live, MiKTeX, or MacTeX) and `latexmk` to compile it.

**Setup Instructions:**
- Install a full LaTeX distribution for your operating system (e.g., `sudo apt install texlive-full latexmk` on Ubuntu/Debian).
- Navigate to the `thesis/` directory and compile the document using:
  ```bash
  cd thesis
  latexmk -pdf thesis_template.tex
  ```

### Number formatting

The thesis and generated figures share `thesis/style/number_formatting.tex`, which
configures **siunitx v3** to print decimal commas and thousands dots
(for example, `\num{1234.56}` renders as **1.234,56**). Install `siunitx` with your
LaTeX distribution if it is not already available.

- Enter numeric quantities with ungrouped, decimal-dot input: `\num{1034}`,
  `\num{0.598}`, `\qty{95}{\percent}`, or `\qty{1.25}{\second}`.
- Use `\numrange{1.04}{1.12}` or `\qtyrange{5}{6}{\percent}` for ranges.
  Separate bracketed confidence-interval endpoints with a semicolon, so decimal
  commas cannot be confused with the endpoint separator.
- Keep model/version identifiers, code literals, URLs, and automatic section,
  reference, and bibliography numbering in their original form. For a displayed
  year, use `\num[group-digits=none]{2026}` to suppress thousands grouping.

The table generators emit siunitx commands, and the figure generators render their
numbers through LaTeX's PGF backend using the same configuration. To regenerate
both, run `uv run --extra visualizations thesis-visualize-all` from `code/`.

## Repository Structure

- **/code**: Contains the Python implementation of the LLM benchmarking framework, data processing scripts, and raw benchmark results.
- **/thesis**: Contains the LaTeX source code, chapters, styling, and figures for the final Master's thesis document.
- **/thesis-proposal**: Includes the initial topic proposal documents and associated references.
- **/literature**: A collection of PDF research papers and surveys referenced throughout the thesis.
- **/declaration**: Contains formal declarations and administrative LaTeX/PDF files.
