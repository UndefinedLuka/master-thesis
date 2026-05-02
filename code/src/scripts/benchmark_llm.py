import json
from pathlib import Path
from typing import Any, List, Optional, TypedDict, cast

import polars as pl
from tqdm import tqdm

from helpers.llm_helpers import LLMMetrics, benchmark_chat_completion, get_llm_model
from helpers.path_helpers import base_path
from helpers.prompt_helpers import get_context


class SpiderResult(TypedDict):
    question: str
    db_id: str
    query: str
    answer: List[dict[str, Any]]
    error: Optional[str]


def load_spider_benchmark(file_path: Path | str) -> List[SpiderResult]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"No file found at {path}")

    with path.open("r", encoding="utf-8") as f:
        data: List[SpiderResult] = json.load(f)

    return data


def load_metrics(file_path: Path | str) -> list[LLMMetrics]:
    path = Path(file_path)

    if not path.exists():
        return []

    print("Sfasdfasd", path)

    df = pl.read_ndjson(path)

    metrics = cast(list[LLMMetrics], df.to_dicts())

    return metrics


class BanchmarkResult(SpiderResult):
    result: LLMMetrics


def save_results(file_path: Path | str, llm_answer: BanchmarkResult):
    path = Path(file_path)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(llm_answer) + "\n")


def run_benchmark(file_path: str, model_name: str, format: str):
    records = load_spider_benchmark(file_path=file_path)
    llm = get_llm_model(model_name)

    print(f"Loaded {len(records)} queries from {file_path}")

    # Load or create progress
    progress_path = Path(
        base_path, "benchmark_results", f"{model_name.replace('.', '-')}_{format}.json"
    )
    last_processed_index = len(load_metrics(progress_path))

    pbar = tqdm(
        records[last_processed_index:],
        total=len(records),
        initial=last_processed_index,
        desc=f"🤖 {model_name[:15]} ({format})",
        unit="query",
        colour="green",
    )

    for item in pbar:
        system_prompt = get_context(db_id=item["db_id"], format=format)
        result = benchmark_chat_completion(llm, system_prompt, item["question"])
        llm_answer: BanchmarkResult = {**item, "result": result}
        save_results(file_path=progress_path, llm_answer=llm_answer)


if __name__ == "__main__":
    run_benchmark(
        Path(base_path, "spider_results.json").as_posix(),
        model_name="gemma-4-E4B-it-Q8_0.gguf",
        format="json",
    )
