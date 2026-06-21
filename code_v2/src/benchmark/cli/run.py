"""Run one shard of the experiment: a single (model, dataset, pipeline[, format]).

For the direct pipeline a ``--format`` is required (one shard per format). For
the sql pipeline ``--format`` is ignored. Results stream to an NDJSON shard and
the run resumes by question index, so it is safe to re-invoke after an interrupt.
"""

from __future__ import annotations

import argparse
import sys

from tqdm import tqdm

from benchmark import paths
from benchmark.config import Config, load_config, with_overrides
from benchmark.datasets import load_questions
from benchmark.llm_client import LLMClient
from benchmark.pipelines.direct import run_direct_question
from benchmark.pipelines.text_to_sql import run_sql_question
from benchmark.reference import load_references
from benchmark.results import ResultRow, already_done_indices, append_row
from benchmark.serialize import read_serialized


def _wait_for_server(client: LLMClient, attempts: int, delay: float) -> bool:
    import time

    for _ in range(attempts):
        if client.health():
            return True
        time.sleep(delay)
    return False


def run_shard(
    cfg: Config,
    model_name: str,
    dataset: str,
    pipeline: str,
    fmt: str | None,
) -> None:
    spec = cfg.dataset(dataset)
    model = cfg.model(model_name)
    questions = load_questions(spec)

    ref_path = paths.reference_file(dataset)
    references = load_references(ref_path) if ref_path.exists() else {}
    if not references:
        print(f"WARNING: no references at {ref_path}; accuracy will be unavailable.")

    shard = paths.results_shard(model_name, dataset, pipeline, fmt)
    done = already_done_indices(shard)

    client = LLMClient(
        base_url=cfg.base_url,
        model=model.model_id,
        api_key=cfg.api_key,
        timeout=cfg.request_timeout_s,
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
    )
    if not _wait_for_server(client, attempts=60, delay=5.0):
        print(f"ERROR: server at {cfg.base_url} not reachable.", file=sys.stderr)
        sys.exit(2)

    label = f"{model_name} {dataset} {pipeline}" + (f"/{fmt}" if fmt else "")
    pbar = tqdm(questions, desc=label, unit="q", initial=0)
    for q in pbar:
        if q.index in done:
            continue
        ref = references.get(q.index)
        db_dir = paths.processed_db_dir(dataset, q.db_id)
        try:
            if pipeline == "direct":
                serialized = read_serialized(db_dir, q.db_id, fmt)  # type: ignore[arg-type]
                row = run_direct_question(
                    client, model_name, dataset, fmt, q, serialized, ref  # type: ignore[arg-type]
                )
            else:
                schema = read_serialized(db_dir, q.db_id, "sql")
                row = run_sql_question(
                    client,
                    model_name,
                    dataset,
                    q,
                    schema,
                    spec.question_db_path(q.db_id),
                    ref,
                    retry_cap=cfg.sql_retry_cap,
                )
        except FileNotFoundError as e:
            row = ResultRow(
                model=model_name,
                dataset=dataset,
                pipeline=pipeline,
                format=fmt,
                question_index=q.index,
                db_id=q.db_id,
                question=q.question,
                error=f"missing serialized artifact: {e}",
            )
        append_row(shard, row)

    print(f"[done] {label} -> {shard}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one benchmark shard.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", required=True, help="model name or model_id from config")
    ap.add_argument("--dataset", required=True, choices=["spider", "bird"])
    ap.add_argument("--pipeline", required=True, choices=["direct", "sql"])
    ap.add_argument("--format", default=None, help="serialization format (direct pipeline)")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-output-tokens", type=int, default=None)
    args = ap.parse_args()

    cfg = with_overrides(
        load_config(args.config),
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
    )

    if args.pipeline == "direct" and not args.format:
        ap.error("--format is required for the direct pipeline")

    run_shard(
        cfg,
        model_name=args.model,
        dataset=args.dataset,
        pipeline=args.pipeline,
        fmt=args.format if args.pipeline == "direct" else None,
    )


if __name__ == "__main__":
    main()
