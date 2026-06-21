"""Generate synthetic benchmark results for previewing the figure suite.

Emits NDJSON shards in the exact :class:`benchmark.results.ResultRow` schema so
``aggregate`` and ``visualize`` can be exercised before the real FRIDA runs
exist. The synthetic numbers encode plausible structure (verbose formats cost
more tokens; larger models score higher; SQL has retries and some failures) so
the rendered figures are representative — they are NOT real measurements.

    uv run python scripts/generate_mock_results.py
"""

from __future__ import annotations

import argparse
import random

from benchmark import paths
from benchmark.config import load_config
from benchmark.results import ResultRow, append_row

# Per-format flavor: (accuracy bias, token multiplier). TOON is terse but newer
# (slight accuracy penalty); XML is verbose; markdown/csv are compact.
FORMAT_PROFILE = {
    "json": (0.02, 1.20),
    "xml": (0.00, 1.55),
    "yaml": (0.03, 1.05),
    "csv": (0.01, 0.80),
    "markdown": (0.02, 0.90),
    "toon": (-0.02, 0.70),
}


def _model_skill(name: str, params: dict[str, float]) -> float:
    """Map parameter count to a base accuracy in ~[0.55, 0.9]."""
    p = params.get(name, 5.0)
    return min(0.9, 0.5 + 0.13 * (p ** 0.5))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def generate(n_questions: int, seed: int) -> None:
    rng = random.Random(seed)
    cfg = load_config()
    # Reuse the viz param map for realistic scaling if present.
    try:
        from benchmark.viz_config import load_viz_config

        params = load_viz_config().model_params_b
    except Exception:
        params = {}

    out_root = paths.results_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    # Clear any previous mock shards for a clean preview.
    for f in out_root.glob("*.ndjson"):
        f.unlink()

    datasets = list(cfg.datasets) or ["spider", "bird"]
    for model in cfg.models:
        skill = _model_skill(model.name, params)
        for dataset in datasets:
            difficulty = 0.0 if dataset == "spider" else -0.12  # BIRD is harder
            db_pool = [f"{dataset}_db_{i}" for i in range(5)]

            # Direct pipeline: one shard per format.
            for fmt in cfg.formats:
                bias, tok_mult = FORMAT_PROFILE.get(fmt, (0.0, 1.0))
                shard = paths.results_shard(model.name, dataset, "direct", fmt)
                for qi in range(n_questions):
                    base = _clip(skill + bias + difficulty + rng.gauss(0, 0.06))
                    in_tok = int(rng.uniform(1500, 4000) * tok_mult)
                    out_tok = int(rng.uniform(15, 90))
                    ttft = rng.uniform(0.15, 0.6) * (in_tok / 2000)
                    append_row(
                        shard,
                        ResultRow(
                            model=model.name, dataset=dataset, pipeline="direct",
                            format=fmt, question_index=qi, db_id=rng.choice(db_pool),
                            question=f"mock question {qi}",
                            output_text="x", reference_text="x",
                            input_tokens=in_tok, output_tokens=out_tok,
                            ttft_s=round(ttft, 4),
                            total_generation_s=round(ttft + out_tok / rng.uniform(20, 45), 4),
                            rouge_l=round(_clip(base - 0.05 + rng.gauss(0, 0.04)), 4),
                            bertscore=round(base, 4),
                        ),
                    )

            # SQL pipeline: schema-only, retries + occasional failure.
            shard = paths.results_shard(model.name, dataset, "sql", None)
            for qi in range(n_questions):
                # Higher-skill models retry less and fail less.
                fail_p = _clip(0.18 - 0.02 * params.get(model.name, 5.0) - difficulty * 0.5, 0.02, 0.4)
                failed = rng.random() < fail_p
                retries = rng.choices([0, 1, 2, 3], weights=[6, 3, 2, 1])[0]
                if failed:
                    retries = 3
                exec_ok = not failed
                match = exec_ok and (rng.random() < _clip(skill + difficulty + 0.05))
                in_tok = int(rng.uniform(400, 1200) * (1 + retries))
                out_tok = int(rng.uniform(30, 140) * (1 + retries))
                ttft = rng.uniform(0.1, 0.4)
                exec_s = rng.uniform(0.002, 0.05)
                base = _clip(skill + difficulty + (0.15 if match else -0.3) + rng.gauss(0, 0.05))
                append_row(
                    shard,
                    ResultRow(
                        model=model.name, dataset=dataset, pipeline="sql",
                        format=None, question_index=qi, db_id=rng.choice(db_pool),
                        question=f"mock question {qi}",
                        output_text="x" if exec_ok else "", reference_text="x",
                        input_tokens=in_tok, output_tokens=out_tok,
                        ttft_s=round(ttft, 4),
                        total_generation_s=round(ttft + out_tok / 35 + exec_s, 4),
                        rouge_l=round(base if exec_ok else 0.0, 4),
                        bertscore=round(base if exec_ok else 0.0, 4),
                        final_sql="SELECT 1",
                        retry_count=retries, definitive_failure=failed,
                        execution_accuracy=exec_ok, result_set_match=match,
                        exec_time_s=round(exec_s, 5),
                        error=None if exec_ok else "no such table",
                    ),
                )

    print(f"Mock results written to {out_root}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate mock benchmark results.")
    ap.add_argument("--n-questions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.n_questions, args.seed)


if __name__ == "__main__":
    main()
