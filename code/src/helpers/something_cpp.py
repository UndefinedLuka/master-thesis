import glob
import time
from pathlib import Path

from llama_cpp import Llama


def run_inference_with_metrics(model: Llama, system_context, question):

    start_time = time.time()
    first_token_time = None
    generated_text = ""

    # We MUST use stream=True to capture the Time To First Token (TTFT)
    stream = model.create_chat_completion(
        messages=[
            {"role": "system", "content": system_context},
            {"role": "user", "content": question},
        ],
        stream=True,
    )

    # Process the stream
    for chunk in stream:
        # Capture the timestamp the moment the very first chunk arrives
        if first_token_time is None:
            first_token_time = time.time()

        # Extract the text chunk
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            text_chunk = delta["content"]
            print(text_chunk, end="", flush=True)
            generated_text += text_chunk

    end_time = time.time()

    # --- Calculate Metrics ---

    ttft = first_token_time - start_time if first_token_time else 0
    total_time = end_time - start_time
    generation_time = end_time - first_token_time if first_token_time else 0

    input_string = f"{system_context}\n{question}"
    input_tokens = len(llm.tokenize(input_string.encode("utf-8")))
    output_tokens = len(llm.tokenize(generated_text.encode("utf-8")))

    tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0

    # Compile the final dictionary
    metrics = {
        "Input Tokens": input_tokens,
        "Output Tokens": output_tokens,
        "Total Tokens": input_tokens + output_tokens,
        "Time to First Token (TTFT)": f"{ttft:.4f} seconds",
        "Generation Time": f"{generation_time:.4f} seconds",
        "Total Request Time": f"{total_time:.4f} seconds",
        "Tokens per Second (TPS)": f"{tokens_per_second:.2f} tok/s",
    }

    # Print the logs nicely formatted
    print("=== PERFORMANCE METRICS ===")
    for key, value in metrics.items():
        print(f"{key:<28}: {value}")

    return generated_text, metrics


if __name__ == "__main__":
    model_path = "./models/gemma-4-E4B-it-Q8_0.gguf"

    llm = Llama(model_path=model_path, n_ctx=0, n_gpu_layers=-1)

    def get_prompt_information(table_id: str, format: str):
        base_path = Path(
            Path(__file__).resolve().parent.parent,
            "data",
            "spider_data_processed",
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

    db_id = "real_estate_properties"
    question = "What are the names of properties that are either houses or apartments with more than 1 room?"
    query = 'SELECT property_name FROM Properties WHERE property_type_code  =  "House" UNION SELECT property_name FROM Properties WHERE property_type_code  =  "Apartment" AND room_count  >  1'
    data_json = get_prompt_information(db_id, "xml")

    # SELECT
    #   T1.property_name
    # FROM Properties AS T1
    # INNER JOIN Ref_Property_Types AS T2
    #   ON T1.property_type_code = T2.property_type_code
    # WHERE
    #   T2.property_type_description LIKE '%House%' OR T2.property_type_description LIKE '%Apartment%' AND T1.room_count > 1;

    with open("./helpers/sql_prompt.md") as promt_file:
        SYSTEM_PROMPT = promt_file.read().replace("<data>", data_json)

    USER_QUESTION = question

    print(SYSTEM_PROMPT)

    try:
        response, run_metrics = run_inference_with_metrics(
            model=llm,
            system_context=SYSTEM_PROMPT,
            question=USER_QUESTION,
        )
    except FileNotFoundError:
        print("Error: Could not find the model file at")
    except Exception as e:
        print(f"An error occurred: {e}")
