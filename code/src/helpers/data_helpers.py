import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from helpers.path_helpers import base_path
from helpers.sqlite_helpers import query_sqlite


def get_all_questions(path: str) -> Dict[str, List[Tuple[str, str]]]:
    with open(path, "r") as json_file:
        questions = json.load(json_file)

    questions_per_db: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for question in questions:
        current_tuple = (question["question"], question["query"])
        (questions_per_db[question["db_id"]]).append(current_tuple)

    return questions_per_db


def get_answers(path: str) -> None:
    questions = get_all_questions(path=path)

    benchmark_results = []
    pbar = tqdm(questions.items(), desc="Starting benchmarks")

    for db_id, question_list in pbar:
        pbar.set_description(f"Processing DB: {db_id}")
        for question, query in question_list:
            try:
                df = query_sqlite(
                    path=Path(
                        base_path,
                        "data",
                        "spider_data",
                        "database",
                        db_id,
                        db_id + ".sqlite",
                    ).as_posix(),
                    query=query,
                )

                answer_data = df.to_dicts()

                entry = {
                    "question": question,
                    "db_id": db_id,
                    "query": query,
                    "answer": answer_data,
                }

                benchmark_results.append(entry)
            except Exception as e:
                # Using postfix to show errors without breaking the bar flow
                pbar.set_postfix(last_error=db_id)
                benchmark_results.append(
                    {
                        "question": question,
                        "db_id": db_id,
                        "query": query,
                        "answer": None,
                        "error": str(e),
                    }
                )

    with open(Path(base_path, "spider_results.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4)


def get_processed_data(table_id: str, format: str):
    path_to_data = Path(
        base_path,
        "data",
        "spider_data_processed",
        "test_database",
        table_id,
    )

    if format == "csv":
        csv_paths = glob.glob(Path(path_to_data, "csv", "*.csv").as_posix())
        content = ""
        for path_str in csv_paths:
            with open(path_str, "r", encoding="utf-8") as file:
                content += file.read() + "\n"
    else:
        path = Path(path_to_data, table_id + "." + format)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

    return content
