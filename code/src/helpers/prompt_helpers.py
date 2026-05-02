from pathlib import Path
from typing import Literal

from helpers.data_helpers import get_processed_data
from helpers.path_helpers import base_path


def get_context(
    db_id: str, format: str, prompt_type: Literal["sql", "direct"] = "direct"
):
    data = get_processed_data(db_id, format)

    prompt_file = "direct_answer.md" if prompt_type == "direct" else "sql_answer.md"

    with open(Path(base_path, "src", "prompts", prompt_file)) as f:
        system_prompt = f.read().replace("<data>", data)

    return system_prompt
