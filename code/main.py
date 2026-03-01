import glob
from helpers.circular_foreign_keys import has_circular_foreign_key
from helpers.read_db_data import get_db_foreign_keys


def main():
    sqlite_paths = glob.glob("data/spider_data/**/**/*.sqlite")

    for path in sqlite_paths:
        foreign_keys = get_db_foreign_keys(path=path)
        has_cycle = has_circular_foreign_key(foreign_keys)
        if has_cycle:
            print(has_cycle, path)
            print(foreign_keys)


if __name__ == "__main__":
    main()
