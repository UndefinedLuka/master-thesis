import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, overload

import toml
import toon
import yaml

from helpers.read_db_metadata import get_all_table_names, query_sqlite

def ensure_path_exists(path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)



def get_db_contents(path: str) ->  Dict[str, List[object]]:
    table_names = get_all_table_names(path=path)

    database = {}

    for name in table_names:
        database[name] = query_sqlite(query=f"select * from {name}", path=path).to_dicts()

    return database

def dump_db_to_json(contents: Dict[str, List[object]], save_path:str) -> None:
    ensure_path_exists(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(contents, f, indent=2, ensure_ascii=False)

def dump_db_to_yaml(contents: Dict[str, List[object]], save_path:str) -> None:
    ensure_path_exists(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(contents, f, indent=2)

def dump_db_to_toml(contents: Dict[str, List[object]], save_path:str) -> None:
    ensure_path_exists(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        toml.dump(contents, f)

def dump_db_to_csv(contents: Dict[str, List[object]], save_path:str) -> None:
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for db_name, records in contents.items():
        if(not isinstance(records, list) or len(records) == 0):
            continue
        
        with open(Path(output_path, db_name + ".csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys()) 
            writer.writeheader()
            writer.writerows(records)

def dump_db_to_toon(contents: Dict[str, List[object]], save_path:str) -> None:
    ensure_path_exists(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(toon.encode(contents))
        

def save_db_to_all_formats(path: str, save_path: str) -> None:
    contents = get_db_contents(path=path)
    file_name = Path(path).name.replace(".sqlite", "")
    # dump_db_to_json(save_path=Path(save_path, file_name + ".json").as_posix(), contents=contents)
    # dump_db_to_yaml(save_path=Path(save_path, file_name + ".yaml").as_posix(), contents=contents)
    # dump_db_to_yaml(save_path=Path(save_path, file_name + ".toml").as_posix(), contents=contents)
    dump_db_to_toon(save_path=Path(save_path, file_name + ".toon").as_posix(), contents=contents)

    # dump_db_to_csv(save_path=Path(save_path, "csv").as_posix(), contents=contents)

