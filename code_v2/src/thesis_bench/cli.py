"""Command-line entry point.

Subcommands:
- ``run``            — run the experiment matrix (optionally narrowed by flags).
- ``list-models``    — preflight: which configured models are available in Ollama.
- ``score-bertscore``— optional post-pass adding BERTScore to result NDJSON files.

Uses stdlib ``argparse`` (no extra dependency). The default experiment comes from
``experiments.default_benchmark()``; flags let you narrow it for quick runs
without editing code.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path

from .config import BenchmarkConfig
from .enums import DatasetKind, PipelineKind, SerializationFormat
from .experiments import default_benchmark
from .llm.factory import build_client
from .llm.ollama_client import OllamaClient
from .runner.orchestrator import Orchestrator


def _narrow(config: BenchmarkConfig, args: argparse.Namespace) -> BenchmarkConfig:
    """Apply CLI overrides (subset of formats/datasets/pipelines, question cap)."""
    changes: dict[str, object] = {}
    if args.formats:
        changes["formats"] = tuple(SerializationFormat(f) for f in args.formats)
    if args.datasets:
        wanted = {DatasetKind(d) for d in args.datasets}
        changes["datasets"] = tuple(d for d in config.datasets if d.kind in wanted)
    if args.pipelines:
        changes["pipelines"] = tuple(PipelineKind(p) for p in args.pipelines)
    if args.limit is not None:
        changes["question_limit"] = args.limit
    if args.results_dir is not None:
        changes["results_dir"] = Path(args.results_dir)
    return dataclasses.replace(config, **changes)  # type: ignore[arg-type]


def _cmd_run(args: argparse.Namespace) -> int:
    config = _narrow(default_benchmark(), args)
    Orchestrator(config).run()
    return 0


def _cmd_list_models(args: argparse.Namespace) -> int:
    config = default_benchmark()
    seen: set[str] = set()
    for model in config.models:
        if model.model_id in seen:
            continue
        seen.add(model.model_id)
        client = build_client(model)
        available = (
            client.list_local_models() if isinstance(client, OllamaClient) else []
        )
        status = "available" if model.model_id in available else "MISSING (ollama pull)"
        print(f"{model.model_id:40s} {status}")
    return 0


def _cmd_score_bertscore(args: argparse.Namespace) -> int:
    from .eval.bertscore_pass import score_file

    total = 0
    for raw in args.files:
        path = Path(raw)
        if not path.exists():
            print(f"[skip] {path} does not exist")
            continue
        n = score_file(path, device=args.device)
        total += n
        print(f"scored {n} records in {path}")
    print(f"done: {total} records")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thesis-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the experiment matrix")
    run.add_argument("--formats", nargs="*", choices=[f.value for f in SerializationFormat])
    run.add_argument("--datasets", nargs="*", choices=[d.value for d in DatasetKind])
    run.add_argument("--pipelines", nargs="*", choices=[p.value for p in PipelineKind])
    run.add_argument("--limit", type=int, default=None, help="max questions per db")
    run.add_argument("--results-dir", default=None)
    run.set_defaults(func=_cmd_run)

    lst = sub.add_parser("list-models", help="show which configured models Ollama has")
    lst.set_defaults(func=_cmd_list_models)

    score = sub.add_parser("score-bertscore", help="add BERTScore to result NDJSON files")
    score.add_argument("files", nargs="+", help="NDJSON result files to score in place")
    score.add_argument("--device", default=None, help="torch device (e.g. cpu, cuda, mps)")
    score.set_defaults(func=_cmd_score_bertscore)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
