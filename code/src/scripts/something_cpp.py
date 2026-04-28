import json
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

from tqdm import tqdm

from helpers.data_helpers import get_processed_data
from helpers.llm_helpers import benchmark_chat_completion, get_llm_model
from helpers.path_helpers import base_path


# Define the shape of your input data
class QueryData(TypedDict):
    question: str
    db_id: str
    query: str
    answer: Optional[list[dict[str, Any]]]


def run_benchmark(file_path: str):
    # 1. Load data using Polars for efficiency
    # Using read_json to handle the array of objects
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON: {e}")
        return

    # Convert to a list of TypedDicts for the loop
    records = cast(list[QueryData], raw_data)

    llm = get_llm_model()
    results = []

    print(len(records))

    # 2. Iterate with tqdm progress bar
    for item in tqdm(records, desc="Processing Benchmark", unit="query"):
        db_id = item["db_id"]
        question = item["question"]

        # In case the JSON format is specific, we get the data string
        # Assuming get_processed_data is your custom helper
        data_json = get_processed_data(db_id, "json")

        with open(Path(base_path, "src", "prompts", "direct_answer.md")) as prompt_file:
            system_prompt = prompt_file.read().replace("<data>", data_json)

        try:
            response = benchmark_chat_completion(
                model=llm,
                system_context=system_prompt,
                question=question,
            )

            print(item, response, sep="\n\n")

            results.append(response)
        except Exception as e:
            print(f"Error processing {db_id}: {e}")
            results.append(None)

    return results


# Usage
# base_path = Path("./")
# run_benchmark("results.json", base_path)

if __name__ == "__main__":
    run_benchmark(Path(base_path, "spider_results.json").as_posix())
