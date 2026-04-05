import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import polars as pl

from helpers.read_db_metadata import query_sqlite


def read_json(path: str) -> Any:
    with open(path, "r") as json_file:
        data = json.load(json_file)

    return data


def get_all_questions(path: str) -> Dict[str, List[Tuple[str, str]]]:
    questions = read_json(path=path)

    questions_per_db: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for question in questions:
        current_tuple = (question["question"], question["query"])
        (questions_per_db[question["db_id"]]).append(current_tuple)

    return questions_per_db


def get_answers(path: str) -> None:
    questions = get_all_questions(path=path)

    for db_id, question_list in questions.items():
        for question, query in question_list:
            print(question, db_id, query, sep="\n")
            try:
                df = query_sqlite(
                    path=f"data/spider_data/database/{db_id}/{db_id}.sqlite",
                    query=query,
                )
                pl.Config(
                    tbl_formatting="NOTHING",
                    tbl_hide_column_data_types=True,
                    tbl_hide_dataframe_shape=True,
                    tbl_hide_column_names=True,
                    fmt_float="mixed",
                    float_precision=3,
                )
                print(df)
                print("=" * 25)
            except:  # noqa: E722
                print("!" * 20, "Something went wrong")
