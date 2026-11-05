"""Question-level Chapter 5 cohorts, selected independently of model correctness.

Load ``COHORT_COLUMNS`` in addition to the fields needed by the analyses. The explicit
campaign roster is intentional: inferring a roster from available rows can make a missing
model or format look complete. References are the same stored records used by scoring,
from full Spider and materialized 500-row-cap BIRD databases.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from config.paths import ProjectPaths
from visualizations.config import DATASET_LABELS, DATASET_ORDER, LATEX_ROW_END
from visualizations.latex_numbers import num

RETAINED_MODELS: tuple[str, ...] = (
    "Qwen3.5-0.8B-Q4_K_M.gguf",
    "Qwen3.5-2B-Q4_K_M.gguf",
    "Qwen3.5-4B-Q4_K_M.gguf",
    "Qwen3.5-9B-Q4_K_M.gguf",
    "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
    "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
    "gemma-4-E2B-it-Q4_K_M.gguf",
    "gemma-4-E4B-it-Q4_K_M.gguf",
    "gemma-4-12b-it-Q4_K_M.gguf",
    "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
)
EXCLUDED_MODELS: tuple[str, ...] = ("Qwen3.5-27B-Q4_K_M.gguf",)
DIRECT_FORMATS: tuple[str, ...] = ("csv", "toon", "markdown", "yaml", "json", "xml")
EXPECTED_CONFIGURATIONS = frozenset(
    (model, pipeline, fmt)
    for model in RETAINED_MODELS
    for pipeline, fmt in [*(("direct", fmt) for fmt in DIRECT_FORMATS), ("text_to_sql", None)]
)
DEFAULT_REFERENCES_DIR = ProjectPaths.discover().data_dir
COHORT_COLUMNS: tuple[str, ...] = (
    "dataset", "example_id", "db_id", "model_id", "pipeline", "format", "error",
    "input_tokens", "retry_count", "attempts.attempt_index", "attempts.error",
    "attempts.llm_metrics.error", "attempts.llm_metrics.input_tokens",
)
_KEY = ["dataset", "example_id"]
_CONTEXT_ERROR = re.compile(
    r"context[_ ](?:exceeded|length|window)|exceed.{0,50}(?:context|n_ctx)|"
    r"(?:context|n_ctx).{0,50}(?:exceed|too (?:long|large)|maximum)|"
    r"too many tokens|prompt.{0,30}too long", re.IGNORECASE,
)


@dataclass(frozen=True)
class AnalysisCohorts:
    manifest: pd.DataFrame
    answerable_records: pd.DataFrame
    context_fit_records: pd.DataFrame
    summary: pd.DataFrame
    audit: dict[str, Any]


def _missing(value: Any) -> bool:
    return value is None or (not isinstance(value, (list, dict)) and bool(pd.isna(value)))


def _positive_tokens(value: Any) -> bool:
    return (
        isinstance(value, Real) and not isinstance(value, bool)
        and math.isfinite(value) and value > 0 and value == int(value)
    )


def _context_error(value: Any) -> bool:
    return isinstance(value, str) and _CONTEXT_ERROR.search(value) is not None


def _context_evidence(row: dict[str, Any]) -> tuple[str, str]:
    """Separate actual inference evidence from generated-answer/execution success.

    Direct has no persisted positive precheck flag. Positive input-token accounting with
    no backend error establishes that inference followed its precheck. SQL has no precheck;
    every initial/correction attempt must establish inference without a context error.
    Unrelated generation errors cannot establish feasibility of the attempted request.
    """
    if _context_error(row["error"]):
        return "not_fit", "recorded_context_error"
    if row["pipeline"] == "direct":
        if not _missing(row["error"]):
            return "unknown", "direct_backend_error"
        if not _positive_tokens(row["input_tokens"]):
            return "unknown", "direct_inference_evidence_missing"
        return "fit", "direct_precheck_pass_inferred_from_inference"

    indices = row["attempts.attempt_index"]
    attempt_errors = row["attempts.error"]
    errors = row["attempts.llm_metrics.error"]
    tokens = row["attempts.llm_metrics.input_tokens"]
    # attempts.error also contains SQL execution/extraction errors (including arbitrary
    # SQL identifiers). Only backend error fields establish context rejection.
    if isinstance(errors, list) and any(_context_error(error) for error in errors):
        return "not_fit", "sql_attempt_context_error"
    if not _missing(row["error"]):
        return "unknown", "sql_backend_error"
    if not all(isinstance(values, list) for values in (indices, attempt_errors, errors, tokens)):
        return "unknown", "sql_attempt_evidence_missing"
    n = len(indices)
    if (
        n not in range(1, 5) or indices != list(range(n))
        or any(len(values) != n for values in (attempt_errors, errors, tokens))
        or _missing(row["retry_count"]) or row["retry_count"] != n - 1
    ):
        return "unknown", "sql_attempt_metadata_inconsistent"
    if any(not _missing(error) for error in errors):
        return "unknown", "sql_attempt_backend_error"
    if not all(_positive_tokens(value) for value in tokens):
        return "unknown", "sql_attempt_inference_evidence_missing"
    if not _positive_tokens(row["input_tokens"]) or sum(tokens) != row["input_tokens"]:
        return "unknown", "sql_input_token_total_inconsistent"
    return "fit", "all_sql_attempts_have_inference_evidence"


def _load_references(directory: Path) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    references: dict[tuple[str, str], dict] = {}
    provenance = []
    for dataset in DATASET_ORDER:
        path = directory / f"{dataset}_references.ndjson"
        if not path.is_file():
            provenance.append({"dataset": dataset, "path": str(path), "status": "missing"})
            continue
        contents = path.read_bytes()
        provenance.append({
            "dataset": dataset, "path": str(path), "status": "present",
            "sha256": hashlib.sha256(contents).hexdigest(),
        })
        for line_number, line in enumerate(contents.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Malformed reference at {path}:{line_number}") from exc
            if not isinstance(record, dict) or not all(
                isinstance(record.get(field), str) and record[field]
                for field in ("example_id", "db_id")
            ):
                raise ValueError(f"Invalid reference identity at {path}:{line_number}")
            if record.get("dataset", dataset) != dataset:
                raise ValueError(f"Reference dataset mismatch at {path}:{line_number}")
            key = (dataset, record["example_id"])
            if key in references:
                raise ValueError(f"Duplicate reference identity: {key}")
            references[key] = record
    return references, provenance


def _reference_status(reference: dict | None) -> str:
    if reference is None:
        return "missing"
    if reference.get("error") is not None:
        return "sql_error"
    count, rows = reference.get("row_count"), reference.get("rows")
    if (
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        or not isinstance(rows, list) or count != len(rows)
        or any(not isinstance(row, dict) for row in rows)
    ):
        return "inconsistent_rows"
    return "nonempty" if count else "empty"


def build_analysis_cohorts(
    results: pd.DataFrame,
    references_dir: Path = DEFAULT_REFERENCES_DIR,
) -> AnalysisCohorts:
    """Audit the retained matrix and select both shared question-level populations.

    Duplicate/invalid identities and database conflicts raise rather than choosing a row.
    Missing configurations or reference evidence remain visible as unknown eligibility;
    callers must check the audit before publishing incomplete campaigns. A known context
    rejection is ``not_fit`` even if another configuration is missing. Reference status
    never affects context eligibility, and context status never affects answerability.
    """
    missing = set(COHORT_COLUMNS).difference(results.columns)
    if missing:
        raise ValueError(f"Missing cohort load columns: {', '.join(sorted(missing))}")
    data = results.copy()
    identities = [*_KEY, "db_id", "model_id", "pipeline"]
    if data[identities].isna().any().any() or any(
        not isinstance(value, str) or not value for column in identities for value in data[column]
    ):
        raise ValueError("Missing or invalid result identity")
    unknown_models = set(data["model_id"]) - set(RETAINED_MODELS) - set(EXCLUDED_MODELS)
    if unknown_models:
        raise ValueError(f"Unexpected model identity: {sorted(unknown_models)}")
    excluded_count = int(data["model_id"].isin(EXCLUDED_MODELS).sum())
    data = data.loc[data["model_id"].isin(RETAINED_MODELS)].copy()
    if not set(data["dataset"]).issubset(DATASET_ORDER):
        raise ValueError("Unexpected dataset identity")
    valid_method = (
        (data["pipeline"].eq("direct") & data["format"].isin(DIRECT_FORMATS))
        | (data["pipeline"].eq("text_to_sql") & data["format"].isna())
    )
    if not valid_method.all():
        raise ValueError("Unexpected pipeline/format configuration")
    identity_columns = [*_KEY, "model_id", "pipeline", "format"]
    if data.duplicated(identity_columns).any():
        raise ValueError("Duplicate model-question-configuration records")
    if data.groupby(_KEY)["db_id"].nunique().gt(1).any():
        raise ValueError("Result database identity conflict")

    references, provenance = _load_references(Path(references_dir))
    grouped = {key: group for key, group in data.groupby(_KEY, sort=True)}
    keys = sorted(set(grouped) | set(references))
    manifest_rows = []
    evidence_totals: Counter = Counter()
    for key in keys:
        group = grouped.get(key, data.iloc[0:0])
        reference = references.get(key)
        db_id = group["db_id"].iloc[0] if len(group) else reference["db_id"]
        if reference is not None and db_id != reference["db_id"]:
            raise ValueError(f"Reference/result database identity mismatch: {key}")
        ref_status = _reference_status(reference)
        configurations = {
            (row.model_id, row.pipeline, None if _missing(row.format) else row.format)
            for row in group.itertuples()
        }
        absent = EXPECTED_CONFIGURATIONS - configurations
        evidence = [_context_evidence(row) for row in group.to_dict("records")]
        statuses = Counter(status for status, _ in evidence)
        reasons = Counter(reason for _, reason in evidence)
        evidence_totals.update(reasons)
        context_status = (
            "not_fit" if statuses["not_fit"] else
            "unknown" if absent or statuses["unknown"] else "fit"
        )
        exclusions = sorted({reason for status, reason in evidence if status != "fit"})
        if absent:
            exclusions.append("missing_configurations")
        manifest_rows.append({
            "dataset": key[0], "example_id": key[1], "db_id": db_id,
            "answerable": ref_status == "nonempty",
            "answerable_status": (
                "eligible" if ref_status == "nonempty" else
                "excluded_empty" if ref_status == "empty" else "unknown"
            ),
            "reference_status": ref_status,
            "reference_row_count": reference.get("row_count") if reference else None,
            "reference_error": reference.get("error") if reference else None,
            "context_fit": context_status == "fit", "context_status": context_status,
            "expected_records": len(EXPECTED_CONFIGURATIONS), "observed_records": len(group),
            "missing_records": len(absent), "fit_records": statuses["fit"],
            "not_fit_records": statuses["not_fit"], "unknown_records": statuses["unknown"],
            "context_reasons": ";".join(exclusions),
            "context_evidence": json.dumps(dict(sorted(reasons.items())), sort_keys=True),
            "missing_configurations": json.dumps(sorted(
                f"{model}|{pipeline}|{fmt or '-'}" for model, pipeline, fmt in absent
            )),
        })
    if not manifest_rows:
        raise ValueError("No question identities in results or references")
    manifest = pd.DataFrame(manifest_rows)
    manifest["reference_row_count"] = pd.array(manifest["reference_row_count"], dtype="Int64")
    row_keys = pd.MultiIndex.from_frame(data[_KEY])

    def selected(flag: str) -> pd.DataFrame:
        selected_keys = pd.MultiIndex.from_frame(manifest.loc[manifest[flag], _KEY])
        return data.loc[row_keys.isin(selected_keys)].copy()

    answerable, context_fit = selected("answerable"), selected("context_fit")
    summary_rows = []
    for dataset in DATASET_ORDER:
        questions = manifest.loc[manifest["dataset"].eq(dataset)]
        if questions.empty:
            continue
        a, c = questions["answerable"], questions["context_fit"]
        summary_rows.append({
            "dataset": dataset, "questions": len(questions),
            "databases": questions["db_id"].nunique(),
            "observations": int(questions["observed_records"].sum()),
            "answerable_questions": int(a.sum()),
            "answerable_databases": questions.loc[a, "db_id"].nunique(),
            "answerable_observations": int(answerable["dataset"].eq(dataset).sum()),
            "empty_references": int(questions["reference_status"].eq("empty").sum()),
            "reference_sql_errors": int(questions["reference_status"].eq("sql_error").sum()),
            "missing_references": int(questions["reference_status"].eq("missing").sum()),
            "inconsistent_references": int(
                questions["reference_status"].eq("inconsistent_rows").sum()
            ),
            "answerable_unknown_questions": int(
                questions["answerable_status"].eq("unknown").sum()
            ),
            "context_fit_questions": int(c.sum()),
            "context_fit_databases": questions.loc[c, "db_id"].nunique(),
            "context_fit_observations": int(context_fit["dataset"].eq(dataset).sum()),
            "context_not_fit_questions": int(questions["context_status"].eq("not_fit").sum()),
            "context_unknown_questions": int(questions["context_status"].eq("unknown").sum()),
            "incomplete_questions": int(questions["missing_records"].gt(0).sum()),
            "missing_records": int(questions["missing_records"].sum()),
            "intersection_questions": int((a & c).sum()),
            "context_fit_empty_reference_questions": int(
                (c & questions["reference_status"].eq("empty")).sum()
            ),
        })
    summary = pd.DataFrame(summary_rows)
    audit = {
        "retained_models": list(RETAINED_MODELS), "excluded_models": list(EXCLUDED_MODELS),
        "direct_formats": list(DIRECT_FORMATS), "expected_records_per_question": 70,
        "excluded_model_records": excluded_count, "retained_records": len(data),
        "duplicate_records": 0, "identity_conflicts": 0,
        "reference_sources": provenance, "datasets": summary.to_dict("records"),
        "context_record_evidence": dict(sorted(evidence_totals.items())),
        "recorded_context_criterion": (
            "Direct: cli/runner.py checks tokenized rendered fixed system prompt (template + "
            "serialized data), plus output headroom, against n_ctx; rejects >= n_ctx. It does "
            "not count the question-bearing chat prompt. Passing is inferred only from "
            "recorded inference. SQL: no runner precheck; require inference evidence for all "
            "recorded initial/correction attempts, no context/backend errors, consistent "
            "attempt indices, retry count and input-token sum. SQL execution/extraction "
            "failures and empty generated output do not affect membership."
        ),
        "chapter_4_handoff": (
            "Chapter 4 describes a complete-prompt check; the runner checks the fixed "
            "system prompt plus headroom. Chapter 5 reports the recorded criterion."
        ),
        "measurement_validation": (
            "Membership is not a timing-validity filter. Analysis consumers must validate "
            "all operational measurements and stop on missing/invalid values."
        ),
    }
    return AnalysisCohorts(manifest, answerable, context_fit, summary, audit)


def generate_cohort_artifacts(
    cohorts: AnalysisCohorts, tables_dir: Path, audit_dir: Path,
) -> list[Path]:
    """Write the compact thesis table and separate reproducible cohort evidence files."""
    tables_dir, audit_dir = Path(tables_dir), Path(audit_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    table = tables_dir / "analysis_cohorts.tex"
    manifest = audit_dir / "analysis_cohort_membership.csv"
    audit = audit_dir / "analysis_cohort_audit.json"
    lines = [
        "% Generated by visualizations.cohorts; do not edit values manually.",
        r"\begin{table}[htbp]", r"  \centering",
        r"  \caption{Analysis cohorts and counts representing "
        r"unique questions and represented databases.}",
        r"  \label{tab:analysis-cohorts}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{@{}llrrr@{}}", r"    \toprule",
        "    Dataset & Cohort & Questions & Databases & Excluded empty references" + LATEX_ROW_END,
        r"    \midrule",
    ]
    for row in cohorts.summary.to_dict("records"):
        label = DATASET_LABELS[row["dataset"]]
        for prefix, name, empty in (
            ("answerable", "Answerable", num(row["empty_references"])),
            ("context_fit", "Context-fit", "---"),
        ):
            lines.append(
                f"    {label} & {name} & {num(row[f'{prefix}_questions'])} & "
                f"{num(row[f'{prefix}_databases'])} & {empty}" + LATEX_ROW_END
            )
    lines.extend([
        r"    \bottomrule", r"  \end{tabular}%", r"  }",
        r"\end{table}", "",
    ])
    table.write_text("\n".join(lines), encoding="utf-8")
    cohorts.manifest.to_csv(manifest, index=False)
    audit.write_text(json.dumps(cohorts.audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return [table, manifest, audit]
