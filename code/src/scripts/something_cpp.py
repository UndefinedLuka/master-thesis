from pathlib import Path

from helpers.data_helpers import get_processed_data
from helpers.llm_helpers import benchmark_chat_completion, get_llm_model
from helpers.path_helpers import base_path

if __name__ == "__main__":
    llm = get_llm_model()

    db_id = "music_1"
    question = "What are the names of the different artists from Bangladesh who never received a rating higher than a 7?"
    query = 'SELECT property_name FROM Properties WHERE property_type_code  =  "House" UNION SELECT property_name FROM Properties WHERE property_type_code  =  "Apartment" AND room_count  >  1'
    data_json = get_processed_data(db_id, "json")

    with open(Path(base_path, "src", "prompts", "sql_answer.md")) as promt_file:
        SYSTEM_PROMPT = promt_file.read().replace("<data>", data_json)

    USER_QUESTION = question

    print(SYSTEM_PROMPT)

    resposne = benchmark_chat_completion(
        model=llm,
        system_context=SYSTEM_PROMPT,
        question=USER_QUESTION,
    )

    print(resposne)
