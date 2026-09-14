"""Shared publication labels, ordering, colors, filenames, and experiment constants."""

from __future__ import annotations

CONTEXT_WINDOW_TOKENS = 131_072
LATEX_ROW_END = " \\\\"

DATASET_ORDER: tuple[str, ...] = ("spider", "bird")
DATASET_LABELS: dict[str, str] = {"spider": "Spider", "bird": "BIRD"}
DATASET_COLORS: dict[str, str] = {"spider": "#0072B2", "bird": "#D55E00"}
EXPECTED_DATABASE_COUNTS: dict[str, int] = {"spider": 20, "bird": 11}
EXPECTED_QUESTION_COUNTS: dict[str, int] = {"spider": 1_034, "bird": 1_534}
DEFAULT_EXCLUDED_MODEL_MARKERS: tuple[str, ...] = ("Qwen3.5-27B",)

SERIALIZATION_FORMAT_ORDER: tuple[str, ...] = (
    "csv",
    "toon",
    "markdown",
    "yaml",
    "json",
    "xml",
)
FORMAT_LABELS: dict[str, str] = {
    "csv": "CSV",
    "toon": "TOON",
    "markdown": "Markdown",
    "yaml": "YAML",
    "json": "JSON",
    "xml": "XML",
}
FORMAT_COLORS: dict[str, str] = {
    "csv": "#0072B2",
    "toon": "#009E73",
    "markdown": "#E69F00",
    "yaml": "#CC79A7",
    "json": "#D55E00",
    "xml": "#56B4E9",
}

SQL_EXECUTION_LABEL = "SQL execution"
METHOD_ORDER: tuple[str, ...] = (
    "csv",
    "markdown",
    "toon",
    "yaml",
    "json",
    "xml",
    SQL_EXECUTION_LABEL,
)
METHOD_LABELS: dict[str, str] = {
    "csv": "CSV",
    "markdown": "Markdown",
    "toon": "TOON",
    "yaml": "YAML",
    "json": "JSON",
    "xml": "XML",
    SQL_EXECUTION_LABEL: "SQL",
}

TOKENIZER_FAMILY_ORDER: tuple[str, ...] = ("qwen-3.5", "ministral-3", "gemma-4")
TOKENIZER_FAMILY_LABELS: dict[str, str] = {
    "qwen-3.5": "Qwen 3.5",
    "ministral-3": "Ministral 3",
    "gemma-4": "Gemma 4",
}
TOKENIZER_FAMILY_COLORS: dict[str, str] = {
    "qwen-3.5": "#0072B2",
    "ministral-3": "#E69F00",
    "gemma-4": "#009E73",
}

METRIC_TITLES: dict[str, str] = {
    "normalized_exact_match": "Normalized Exact Match",
    "rouge_l_f": "ROUGE-L F1",
    "bertscore_f": "BERTScore F1",
}
DIRECT_ACCURACY_METRIC_ORDER: tuple[str, ...] = (
    "normalized_exact_match",
    "bertscore_f",
    "rouge_l_f",
)

OUTCOME_LABELS: dict[str, str] = {
    "success_clean": "Successful without a recorded error",
    "success_after_correction": "Successful after SQL error correction",
    "failed_context": "Failed: context window exceeded",
    "failed_sql": "Failed: SQL execution error after retry limit",
    "failed_no_sql": "Failed: no SQL statement extracted",
    "failed_generation": "Failed: model or backend generation error",
    "failed_direct_output": "Failed: empty or malformed direct output",
    "failed_other": "Failed: other error",
}

EXPERIMENT_OUTCOMES_TABLE_FILENAME = "experiment_outcomes.tex"
EXPERIMENT_OUTCOMES_TABLE_LABEL = "tab:experiment-outcomes"
SERIALIZATION_AVERAGE_TABLE_FILENAME = "serialization_average_tokens.tex"
SERIALIZATION_AVERAGE_TABLE_LABEL = "tab:serialization-average-tokens"
SERIALIZATION_FIT_TABLE_FILENAME = "serialization_database_fit.tex"
SERIALIZATION_FIT_TABLE_LABEL = "tab:serialization-database-fit"
TOKENIZER_COMPARISON_FIGURE_FILENAME = "tokenizer_family_comparison.pdf"
TOKENIZER_COMPARISON_FIGURE_LABEL = "fig:tokenizer-family-comparison"
DIRECT_ACCURACY_FORMAT_FIGURE_FILENAME = "direct_accuracy_by_format.pdf"
DIRECT_ACCURACY_FORMAT_FIGURE_LABEL = "fig:direct-accuracy-by-format"
DIRECT_ACCURACY_MODEL_SIZE_FIGURE_FILENAME = "direct_accuracy_by_model_size.pdf"
DIRECT_ACCURACY_MODEL_SIZE_FIGURE_LABEL = "fig:direct-accuracy-by-model-size"
DIRECT_OPERATIONAL_TABLE_FILENAME = "direct_operational_efficiency.tex"
DIRECT_OPERATIONAL_TABLE_LABEL = "tab:direct-operational-efficiency"
DIRECT_TOKEN_USAGE_FIGURE_FILENAME = "direct_token_usage_by_format.pdf"
DIRECT_TOKEN_USAGE_FIGURE_LABEL = "fig:direct-token-usage-by-format"
DIRECT_LATENCY_FIGURE_FILENAME = "direct_latency_by_format.pdf"
DIRECT_LATENCY_FIGURE_LABEL = "fig:direct-latency-by-format"
DIRECT_DATABASE_SIZE_FIGURE_FILENAMES: dict[str, str] = {
    "normalized_exact_match": "direct_accuracy_by_database_size_nem.pdf",
    "bertscore_f": "direct_accuracy_by_database_size_bertscore.pdf",
    "rouge_l_f": "direct_accuracy_by_database_size_rouge_l.pdf",
}
DIRECT_DATABASE_SIZE_FIGURE_LABELS: dict[str, str] = {
    "normalized_exact_match": "fig:direct-accuracy-by-database-size-nem",
    "bertscore_f": "fig:direct-accuracy-by-database-size-bertscore",
    "rouge_l_f": "fig:direct-accuracy-by-database-size-rouge-l",
}
TEXT_TO_SQL_ACCURACY_FIGURE_FILENAME = "text_to_sql_accuracy_by_model_size.pdf"
TEXT_TO_SQL_ACCURACY_FIGURE_LABEL = "fig:text-to-sql-accuracy-by-model-size"
TEXT_TO_SQL_RETRIEVAL_TABLE_FILENAME = "text_to_sql_retrieval_accuracy.tex"
TEXT_TO_SQL_RETRIEVAL_TABLE_LABEL = "tab:text-to-sql-retrieval-accuracy"
TEXT_TO_SQL_OPERATIONAL_TABLE_FILENAME = "text_to_sql_operational_efficiency.tex"
TEXT_TO_SQL_OPERATIONAL_TABLE_LABEL = "tab:text-to-sql-operational-efficiency"
TEXT_TO_SQL_TOKEN_FIGURE_FILENAME = "text_to_sql_token_usage.pdf"
TEXT_TO_SQL_TOKEN_FIGURE_LABEL = "fig:text-to-sql-token-usage"
TEXT_TO_SQL_LATENCY_FIGURE_FILENAME = "text_to_sql_latency.pdf"
TEXT_TO_SQL_LATENCY_FIGURE_LABEL = "fig:text-to-sql-latency"
TEXT_TO_SQL_RETRY_TABLE_FILENAME = "text_to_sql_error_correction.tex"
TEXT_TO_SQL_RETRY_TABLE_LABEL = "tab:text-to-sql-error-correction"
TEXT_TO_SQL_RETRY_FIGURE_FILENAME = "text_to_sql_error_correction.pdf"
TEXT_TO_SQL_RETRY_FIGURE_LABEL = "fig:text-to-sql-error-correction"
RETRY_OUTCOME_ORDER: tuple[str, ...] = (
    "attempt_1",
    "attempt_2",
    "attempt_3",
    "attempt_4",
    "failure",
)
RETRY_OUTCOME_LABELS: dict[str, str] = {
    "attempt_1": "First attempt",
    "attempt_2": "Second attempt",
    "attempt_3": "Third attempt",
    "attempt_4": "Fourth attempt",
    "failure": "Complete failure",
}


LATENCY_PHASE_LABELS: dict[str, str] = {
    "ttft": "Time to first token",
    "generation": "Generation after first token",
}
LATENCY_PHASE_COLORS: dict[str, str] = {
    "ttft": "#0072B2",
    "generation": "#E69F00",
}
TEXT_TO_SQL_LATENCY_PHASE_LABELS: dict[str, str] = {
    "ttft": "Cumulative first-token waits",
    "generation": "Generation after first tokens",
    "sql": "SQL execution",
}
TEXT_TO_SQL_LATENCY_PHASE_COLORS: dict[str, str] = {
    "ttft": "#0072B2",
    "generation": "#E69F00",
    "sql": "#009E73",
}
