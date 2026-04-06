from pathlib import Path

from helpers.data_helpers import get_answers
from helpers.path_helpers import base_path

if __name__ == "__main__":
    get_answers(Path(base_path, "data", "spider_data", "train_spider.json").as_posix())
    # my_data = get_sql_data()
