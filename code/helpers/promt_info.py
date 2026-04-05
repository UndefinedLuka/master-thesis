import glob
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
from read_db_metadata import query_sqlite


def get_prompt_information(table_id: str, format: str):
    base_path = Path(
        Path(__file__).resolve().parent,
        "..",
        "temp",
        "data",
        "spider_data",
        "test_database",
        table_id,
    )

    if format == "csv":
        csv_paths = glob.glob(Path(base_path, "csv", "*.csv").as_posix())
        content = ""
        for path_str in csv_paths:
            with open(path_str, "r", encoding="utf-8") as file:
                content += file.read() + "\n"
    else:
        path = Path(base_path, table_id + "." + format)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

    return content


def get_sql_data(file_name: str):
    path = Path(__file__).parent.parent / file_name

    if not path.exists():
        print(f"File {file_name} not found.")
        return []

    # 2. Read file and convert to objects using SimpleNamespace
    # The object_hook transforms every {key: value} into an object
    content = path.read_text(encoding="utf-8")
    data = json.loads(content, object_hook=lambda d: SimpleNamespace(**d))

    # 3. Access elements with '.' notation
    for item in data:
        print(f"ID:       {item.db_id}")
        print(f"Question: {item.question}")
        print(f"SQL:      {item.query}")
        print(f"Data:     {len(get_prompt_information(item.db_id, 'toon'))}")
        try:
            df = query_sqlite(
                path=f"data/spider_data/database/{item.db_id}/{item.db_id}.sqlite",
                query=item.query,
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
        print("-" * 30)

    return data


if __name__ == "__main__":
    # Replace 'data.json' with your filename
    my_data = get_sql_data("data/spider_data/dev.json")
