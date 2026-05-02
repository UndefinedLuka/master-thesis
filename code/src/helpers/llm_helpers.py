import time
from pathlib import Path
from typing import Iterator, Optional, TypedDict, cast

from llama_cpp import CreateChatCompletionStreamResponse, Llama

from helpers.path_helpers import base_path

LLM: Llama | None = None


def get_llm_model(model_name="Qwen3.5-9B.Q8_0.gguf"):
    global LLM

    model_path = Path(base_path, "models", model_name).as_posix()

    if LLM is not None:
        return LLM

    try:
        LLM = Llama(
            model_path=model_path,
            n_ctx=126_000,
            n_gpu_layers=0,
            n_threads=8,
        )
    except FileNotFoundError:
        print("Error: Could not find the model file at")
    except Exception as e:
        print(f"An error occurred: {e}")

    if LLM is None:
        raise Exception("Could not Load the specified model")

    return LLM


class LLMMetrics(TypedDict):
    input_tokens: int
    output_tokens: int
    time_to_first_token: float
    total_response_time: float
    tokens_per_second: float
    full_response: str
    stop_reason: Optional[str]
    error: Optional[str]


def benchmark_chat_completion(
    model: Llama, system_context: str, question: str
) -> LLMMetrics:

    metrics: LLMMetrics = {
        "input_tokens": 0,
        "output_tokens": 0,
        "time_to_first_token": 0.0,
        "total_response_time": 0.0,
        "tokens_per_second": 0.0,
        "full_response": "",
        "stop_reason": None,
        "error": None,
    }

    start_time = time.perf_counter()

    try:
        stream = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": question},
            ],
            stream=True,
        )

        first_token_received = False
        token_count = 0

        for chunk in cast(Iterator[CreateChatCompletionStreamResponse], stream):
            if not first_token_received:
                metrics["time_to_first_token"] = time.perf_counter() - start_time
                first_token_received = True

            # Extract text content (structure varies slightly by provider)
            delta = chunk.get("choices", [])[0].get("delta", {})
            content = delta.get("content", "")

            if content:
                metrics["full_response"] += content
                token_count += 1

            if chunk["choices"][0].get("finish_reason"):
                metrics["stop_reason"] = chunk["choices"][0]["finish_reason"]

        # 4. Final Calculations
        end_time = time.perf_counter()
        metrics["total_response_time"] = end_time - start_time

        metrics["input_tokens"] = len(
            model.tokenize(f"{system_context}\n{question}".encode("utf-8"))
        )

        metrics["output_tokens"] = len(metrics["full_response"])

        # Fallback if metadata wasn't provided by the API
        if metrics["output_tokens"] == 0:
            metrics["output_tokens"] = token_count

        # Calculate throughput (Tokens Per Second)
        # We use (Total Time - TTFT) for generation speed, or Total Time for overall speed
        generation_time = (
            metrics["total_response_time"] - metrics["time_to_first_token"]
        )
        if generation_time > 0:
            metrics["tokens_per_second"] = metrics["output_tokens"] / generation_time
        else:
            metrics["tokens_per_second"] = 0.0

    except Exception as e:
        metrics["error"] = str(e)

    return metrics
