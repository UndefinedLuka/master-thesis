"""Run every GGUF model in the models/ directory across all datasets and pipelines.

python -m cli.run_all
python -m cli.run_all --limit 50                       # smoke test: first N examples per dataset
python -m cli.run_all --models Qwen3.5-2B-Q4_K_M.gguf  # just one model

Scans ``models/`` for ``*.gguf`` files and runs each one through both pipelines
(direct + text_to_sql) on both datasets (spider + bird). Each model is loaded once and
reused across all its dataset/pipeline combinations. Results resume automatically —
already-finished slices are skipped.

``--models`` narrows the run to an explicit subset. Since every slice writes to a file named
after its model, one process per model can run concurrently without conflict — which is how
``deploy/run_models.sbatch`` fans the models out across GPUs as a Slurm job array.
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


def _find_models(models_dir: Path, only: list[str] | None = None) -> list[str]:
    """GGUF filenames to run from ``models_dir``; raises if the directory is empty/missing.

    Without ``only``, returns every ``*.gguf`` sorted by name. With ``only``, returns exactly
    those filenames in the order requested, raising if any is absent — a typo must fail the run
    immediately rather than quietly benchmark nothing.
    """
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    available = sorted(p.name for p in models_dir.glob("*.gguf"))
    if not available:
        raise FileNotFoundError(f"No .gguf files found in {models_dir}")
    if only is None:
        return available

    missing = [name for name in only if name not in set(available)]
    if missing:
        raise FileNotFoundError(
            f"Model(s) not found in {models_dir}: {', '.join(missing)}. "
            f"Available: {', '.join(available)}"
        )
    return list(dict.fromkeys(only))  # de-duplicate, preserving the requested order


def _parse_models(raw: str) -> list[str]:
    """Parse the --models value: a comma-separated list of GGUF filenames."""
    names = [part.strip() for part in raw.split(",")]
    names = [name for name in names if name]
    if not names:
        raise argparse.ArgumentTypeError("--models needs at least one filename")
    return names


def main(argv: list[str] | None = None) -> int:
    """Run all models x datasets x pipelines, loading each model once."""
    parser = argparse.ArgumentParser(description="Run all models x datasets x pipelines")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run the first N examples per dataset (smoke tests; full run without it)",
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=None,
        help="comma-separated GGUF filenames to run (default: every *.gguf in models/)",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.discover()
    models = _find_models(paths.models_dir, args.models)
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
