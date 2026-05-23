import csv
import json
from pathlib import Path
from typing import Dict, List

import polars as pl
import tomlkit
import toon
import yaml
from dict2xml import dict2xml
from tqdm import tqdm

from helpers.path_helpers import base_path
from helpers.sqlite_helpers import query_sqlite


def get_all_table_names(path: str) -> List[str]:
    query = """select name from sqlite_master where type = "table" and sql not null;"""

    df = query_sqlite(path=path, query=query)

    return df["name"].to_list()


def ensure_path_exists(path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def get_db_contents(path: str) -> Dict[str, List[Dict]]:
    table_names = get_all_table_names(path=path)

    database = {}

    for name in table_names:
        database[name] = query_sqlite(
            query=f"select * from {name}", path=path
        ).to_dicts()

    return database


def dump_db_schema(path: str, save_path: str) -> None:
    query = "SELECT * FROM sqlite_master WHERE type='table' AND sql IS NOT NULL;"

    df = query_sqlite(path=path, query=query)

    ensure_path_exists(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(df["sql"].to_list()))


def dump_db_to_json(contents: Dict[str, List[Dict]], save_path: str) -> None:
    ensure_path_exists(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(contents, f, indent=2, ensure_ascii=False)


def dump_db_to_yaml(contents: Dict[str, List[Dict]], save_path: str) -> None:
    ensure_path_exists(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(contents, f, indent=2)


def dump_db_to_toml(contents: Dict[str, List[Dict]], save_path: str) -> None:
    ensure_path_exists(save_path)

    def to_toml_compatible(data):
        if isinstance(data, (pl.Series, list)):
            return [to_toml_compatible(item) for item in data]
        elif isinstance(data, (pl.DataFrame, dict)):
            items = data.to_dicts() if isinstance(data, pl.DataFrame) else data.items()
            if isinstance(items, list):
                return [to_toml_compatible(d) for d in items]
            return {str(k): to_toml_compatible(v) for k, v in items}
        elif data is None:
            return "NULL"
        elif hasattr(data, "item"):
            return data.item()
        return data

    toml_string = tomlkit.dumps(to_toml_compatible(contents))  # pyright: ignore[reportArgumentType]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(toml_string)


def dump_db_to_csv(contents: Dict[str, List[Dict]], save_path: str) -> None:
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for db_name, records in contents.items():
        if not isinstance(records, list) or len(records) == 0:
            continue

        with open(Path(output_path, db_name + ".csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)


def dump_db_to_toon(contents: Dict[str, List[Dict]], save_path: str) -> None:
    ensure_path_exists(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(toon.encode(contents))


def dump_db_to_xml(contents: Dict[str, List[Dict]], save_path: str) -> None:
    ensure_path_exists(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(dict2xml(contents, wrap="root", indent="\t"))


def save_db_to_all_formats(path: str, save_path: str) -> None:
    contents = get_db_contents(path=path)
    file_name = Path(path).stem.replace(".sqlite", "")
    dump_db_schema(
        path=path,
        save_path=Path(save_path, file_name + ".sql").as_posix(),
    )
    dump_db_to_json(
        save_path=Path(save_path, file_name + ".json").as_posix(), contents=contents
    )
    dump_db_to_yaml(
        save_path=Path(save_path, file_name + ".yaml").as_posix(), contents=contents
    )
    # dump_db_to_toml(
    #     save_path=Path(save_path, file_name + ".toml").as_posix(), contents=contents
    # )
    dump_db_to_toon(
        save_path=Path(save_path, file_name + ".toon").as_posix(), contents=contents
    )
    dump_db_to_xml(
        save_path=Path(save_path, file_name + ".xml").as_posix(), contents=contents
    )
    dump_db_to_csv(save_path=Path(save_path, "csv").as_posix(), contents=contents)


if __name__ == "__main__":
    data_path = Path(base_path, "data", "spider_data")
    save_path_base = Path(base_path, "data", "spider_data_processed")

    pbar = tqdm(list(data_path.rglob("*.sqlite")))

    for sqlit_path in pbar:
        pbar.set_description(f"Processing {sqlit_path.name}")
        save_path = save_path_base / sqlit_path.relative_to(data_path).parent

        save_db_to_all_formats(
            path=sqlit_path.as_posix(), save_path=save_path.as_posix()
        )
