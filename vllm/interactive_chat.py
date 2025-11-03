import os
from time import perf_counter

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

MODEL_ID = "TechxGenus/gemma-2b-it-AWQ"  # "google/gemma-2-2b-it"

os.environ["VLLM_ATTENTION_BACKEND"] = (
    "FLASH_ATTN"  # Gemma can't handle default attention?
)

llm = LLM(
    model=MODEL_ID,
    dtype="float16",
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    quantization="awq",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

history = []
sampling = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    stop_token_ids=[tokenizer.convert_tokens_to_ids("<end_of_turn>")],
)

while True:
    user_msg = input("You: ").strip()
    if user_msg.lower() in {"exit", "quit"}:
        break

    history.append({"role": "user", "content": user_msg})
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )

    start = perf_counter()
    outputs = llm.generate(prompt, sampling_params=sampling, use_tqdm=False)
    end = perf_counter()

    reply = outputs[0].outputs[0].text
    print(reply)

    tok = len(tokenizer.encode(reply))
    print(f"{tok / (end - start):.2f} tok/s")

    history.append({"role": "assistant", "content": reply})
