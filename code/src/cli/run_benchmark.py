"""Run a benchmark pipeline over a dataset, streaming results to resumable NDJSON.

python -m cli.run_benchmark --dataset spider --pipeline direct --model Qwen2.5-7B-Q8_0.gguf
python -m cli.run_benchmark --dataset spider --pipeline direct --formats json,markdown
python -m cli.run_benchmark --dataset bird  --pipeline text_to_sql --model local
"""

from __future__ import annotations

import argparse
from typing import get_args

from cli.runner import dataset_config_for, run_direct_slices, run_sql_slice
from config.config import BACKEND, BASE_URL
from config.paths import ProjectPaths
from config.settings import (
    BackendConfig,
    DatasetName,
    PipelineKind,
    RunConfig,
)
from datasets.loader import load_dataset
from llm.factory import get_backend
from serialization.types import FORMATS, SerializationFormat


def _parse_formats(raw: str) -> tuple[SerializationFormat, ...]:
    """Parse the --formats value ('all' or a comma-separated subset, order-preserving)."""
    if raw == "all":
        return FORMATS
    valid = set(FORMATS)
    chosen: list[SerializationFormat] = []
    for part in raw.split(","):
        name = part.strip()
        if name not in valid:
            raise argparse.ArgumentTypeError(
                f"Unknown format {name!r}; valid: {', '.join(FORMATS)}, all"
            )
        if name not in chosen:
            chosen.append(name)  # type: ignore[arg-type]
    return tuple(chosen)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, wire up the run configuration, and run one benchmark slice."""
    parser = argparse.ArgumentParser(description="Run a benchmark pipeline")
    parser.add_argument("--dataset", choices=list(get_args(DatasetName)), required=True)
    parser.add_argument("--pipeline", choices=list(get_args(PipelineKind)), required=True)
    parser.add_argument("--model", required=True, help="GGUF filename or served model name")
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=FORMATS,
        help=f"comma-separated subset of {','.join(FORMATS)}, or 'all' (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run the first N examples (smoke tests; full run without it)",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args(argv)

    paths = ProjectPaths.discover()
    dataset = dataset_config_for(args.dataset, paths)
    cfg = RunConfig(
        backend=BackendConfig(kind=BACKEND, model_id=args.model, base_url=BASE_URL),
        dataset=dataset,
        pipeline=args.pipeline,
        formats=args.formats,
        results_dir=paths.results_dir,
        resume=args.resume,
    )

    examples = load_dataset(cfg.dataset)
    if args.limit is not None:
        examples = examples[: args.limit]
        print(f"*** LIMIT ACTIVE: only the first {args.limit} examples ***")
    print(
        f"backend={cfg.backend.kind} model={cfg.backend.model_id} "
        f"dataset={cfg.dataset.name} pipeline={cfg.pipeline}"
    )
    print(f"Loaded {len(examples)} examples; results under {cfg.results_dir}/")

    backend = get_backend(cfg.backend)
    try:
        if cfg.pipeline == "direct":
            run_direct_slices(backend, cfg, examples)
        else:
            run_sql_slice(backend, cfg, examples)
    finally:
        backend.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
