import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

from tqdm import tqdm

from helpers.data_helpers import get_processed_data
from helpers.llm_helpers import LLMMetrics, benchmark_chat_completion, get_llm_model
from helpers.path_helpers import base_path


# Define the shape of your input data
class QueryData(TypedDict):
    question: str
    db_id: str
    query: str
    answer: Optional[list[dict[str, Any]]]


class ProgressItem(TypedDict):
    db_id: str
    question: str
    metrics: LLMMetrics
    attempts: int


class FailedItem(TypedDict):
    db_id: str
    question: str
    error: str
    attempts: int


class ProgressFile(TypedDict):
    version: str
    source_file: str
    total_items: int
    completed: dict[str, ProgressItem]
    failed: dict[str, FailedItem]


def load_progress(progress_path: Path, source_file: str, total: int) -> ProgressFile:
    """Load existing progress or create empty structure."""
    if not progress_path.exists():
        return {
            "version": "1.0",
            "source_file": source_file,
            "total_items": total,
            "completed": {},
            "failed": {},
        }

    try:
        with open(progress_path, encoding="utf-8") as f:
            progress = cast(ProgressFile, json.load(f))
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: Cannot read progress file ({e}). Starting fresh.")
        return load_progress(progress_path, source_file, total)

    # Validate progress matches source data
    if (
        progress.get("source_file") != source_file
        or progress.get("total_items") != total
    ):
        print("WARNING: Progress file doesn't match source data. Starting fresh.")
        return {
            "version": "1.0",
            "source_file": source_file,
            "total_items": total,
            "completed": {},
            "failed": {},
        }

    completed = len(progress["completed"])
    failed = len(progress["failed"])
    remaining = total - completed - failed
    print(f"Resuming: {completed} completed, {failed} failed, {remaining} remaining")
    return progress


def save_progress_atomic(progress: ProgressFile, progress_path: Path) -> None:
    """Save progress using atomic write (temp file + rename)."""
    try:
        temp_path = progress_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
        temp_path.replace(progress_path)  # Atomic on POSIX systems
    except OSError as e:
        print(f"WARNING: Failed to save progress: {e}")
        # Don't crash - continue without checkpoint protection


def retry_with_backoff(func, max_attempts: int = 3, base_delay: float = 2.0):
    """Execute function with exponential backoff retry."""
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = func()
            return result, attempt, None  # Success
        except Exception as e:
            last_error = str(e)
            if attempt < max_attempts:
                wait_time = base_delay**attempt
                print(f"  Attempt {attempt} failed: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"  Attempt {attempt} failed: {e}. Giving up.")

    return None, max_attempts, last_error  # Failed after max attempts


def run_benchmark(file_path: str):
    # Validate source file exists
    source_path = Path(file_path)
    if not source_path.exists():
        print(f"FATAL: Source file not found: {file_path}")
        sys.exit(1)

    # Load and validate JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"FATAL: Invalid JSON in source file: {e}")
        sys.exit(1)

    # Convert to a list of TypedDicts for the loop
    records = cast(list[QueryData], raw_data)

    # Initialize LLM (will raise exception if fails)
    try:
        llm = get_llm_model()
    except Exception as e:
        print(f"FATAL: Cannot initialize LLM model: {e}")
        sys.exit(1)

    print(f"Loaded {len(records)} queries from {file_path}")

    # Load or create progress
    progress_path = Path(base_path, "benchmark_progress.json")
    progress = load_progress(progress_path, file_path, len(records))

    # Set up progress bar
    pbar = tqdm(total=len(records), desc="Benchmarking LLM QA", unit="query")
    # Start at correct position
    pbar.update(len(progress["completed"]) + len(progress["failed"]))

    # Process each item
    for index, item in enumerate(records):
        # Skip already processed
        idx_str = str(index)
        if idx_str in progress["completed"] or idx_str in progress["failed"]:
            continue

        # Define processing function for retry wrapper
        def process_item():
            data_json = get_processed_data(item["db_id"], "json")
            with open(Path(base_path, "src", "prompts", "direct_answer.md")) as f:
                system_prompt = f.read().replace("<data>", data_json)
            return benchmark_chat_completion(llm, system_prompt, item["question"])

        # Execute with retry
        result, attempts, error = retry_with_backoff(process_item, max_attempts=3)

        # Record outcome
        if result is None or result.get("error"):
            final_error = error or result.get("error", "Unknown error")
            progress["failed"][idx_str] = {
                "db_id": item["db_id"],
                "question": item["question"],
                "error": final_error,
                "attempts": attempts,
            }
        else:
            progress["completed"][idx_str] = {
                "db_id": item["db_id"],
                "question": item["question"],
                "metrics": result,
                "attempts": attempts,
            }

        # Save checkpoint
        save_progress_atomic(progress, progress_path)

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix(
            {
                "completed": len(progress["completed"]),
                "failed": len(progress["failed"]),
                "attempts": attempts,
            }
        )

    pbar.close()

    # Rebuild results array in order
    final_results = []
    for i in range(len(records)):
        idx_str = str(i)
        if idx_str in progress["completed"]:
            final_results.append(progress["completed"][idx_str])
        elif idx_str in progress["failed"]:
            final_results.append(
                {
                    "db_id": progress["failed"][idx_str]["db_id"],
                    "question": progress["failed"][idx_str]["question"],
                    "error": progress["failed"][idx_str]["error"],
                    "attempts": progress["failed"][idx_str]["attempts"],
                }
            )
        else:
            # Should never happen
            final_results.append(None)

    # Save final results
    results_path = Path(base_path, "benchmark_results_llm_qa.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Benchmark Complete ===")
    print(f"Results saved to: {results_path}")
    print(f"Successful: {len(progress['completed'])}")
    print(f"Failed: {len(progress['failed'])}")

    # Clean up progress file
    try:
        progress_path.unlink()
        print("Progress file cleaned up.")
    except OSError as e:
        print(f"WARNING: Could not delete progress file: {e}")


if __name__ == "__main__":
    run_benchmark(Path(base_path, "spider_results.json").as_posix())
