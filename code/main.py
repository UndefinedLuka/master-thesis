import glob
from pathlib import Path

from helpers.db_data import save_db_to_all_formats
from helpers.questions import get_answers


def main():
    sqlite_paths = glob.glob("data/spider_data/**/**/*.sqlite")

    for path_str in sqlite_paths:
        path = Path(path_str)
        print("Reading: ", path)

        # foreign_keys = get_db_foreign_keys(path=path.as_posix())
        # print(find_root_node(foreign_keys))
        # print_database_connections(foreign_keys, path)
        # has_cycle = has_circular_foreign_key(foreign_keys)
        save_db_to_all_formats(
            path=path.as_posix(), save_path=Path("temp", path.parent).as_posix()
        )

        # if has_cycle:
        #     print(has_cycle, path)
        #     print(foreign_keys)

    # print(get_all_questions("data/spider_data/dev.json"))

    get_answers("data/spider_data/dev.json")


if __name__ == "__main__":
    main()
