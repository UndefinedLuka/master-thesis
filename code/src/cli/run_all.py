"""Run every GGUF model in the models/ directory across all datasets and pipelines.

python -m cli.run_all
python -m cli.run_all --limit 50   # smoke test: first N examples per dataset

Scans ``models/`` for ``*.gguf`` files and runs each one through both pipelines
(direct + text_to_sql) on both datasets (spider + bird). Each model is loaded once and
reused across all its dataset/pipeline combinations. Results resume automatically —
already-finished slices are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import get_args

from cli.runner import dataset_config_for, run_direct_slices, run_sql_slice
from config.config import BACKEND, BASE_URL
from config.paths import ProjectPaths
from config.settings import (
    BackendConfig,
    DatasetConfig,
    DatasetName,
    PipelineKind,
    RunConfig,
)
from datasets.loader import load_dataset
from datasets.types import DatasetExample
from llm.factory import get_backend


def _find_models(models_dir: Path) -> list[str]:
    """Sorted GGUF filenames in ``models_dir``; raises if the directory is empty/missing."""
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    models = sorted(p.name for p in models_dir.glob("*.gguf"))
    if not models:
        raise FileNotFoundError(f"No .gguf files found in {models_dir}")
    return models


def main(argv: list[str] | None = None) -> int:
    """Run all models x datasets x pipelines, loading each model once."""
    parser = argparse.ArgumentParser(description="Run all models x datasets x pipelines")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run the first N examples per dataset (smoke tests; full run without it)",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.discover()
    models = _find_models(paths.models_dir)
    dataset_names = get_args(DatasetName)
    pipelines = get_args(PipelineKind)

    print(f"Found {len(models)} model(s) in {paths.models_dir}:")
    for m in models:
        print(f"  {m}")
    if args.limit is not None:
        print(f"\n*** LIMIT ACTIVE: only the first {args.limit} examples per dataset ***\n")

    datasets: dict[DatasetName, tuple[DatasetConfig, list[DatasetExample]]] = {}
    for name in dataset_names:
        dataset = dataset_config_for(name, paths)
        examples = load_dataset(dataset)
        if args.limit is not None:
            examples = examples[: args.limit]
        datasets[name] = (dataset, examples)
        print(f"{name}: {len(examples)} examples")

    total = len(models) * len(datasets) * len(pipelines)
    done = 0

    for model in models:
        # One backend (one model load) per GGUF, reused across datasets and pipelines.
        backend_cfg = BackendConfig(kind=BACKEND, model_id=model, base_url=BASE_URL)
        backend = get_backend(backend_cfg)
        try:
            for dataset_name, (dataset, examples) in datasets.items():
                for pipeline in pipelines:
                    done += 1
                    print(
                        f"\n[{done}/{total}] model={model}"
                        f" dataset={dataset_name} pipeline={pipeline}"
                    )
                    cfg = RunConfig(
                        backend=backend_cfg,
                        dataset=dataset,
                        pipeline=pipeline,
                        results_dir=paths.results_dir,
                    )
                    if pipeline == "direct":
                        run_direct_slices(backend, cfg, examples)
                    else:
                        run_sql_slice(backend, cfg, examples)
        finally:
            backend.close()

    print("\nAll runs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
