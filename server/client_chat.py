import os
from time import perf_counter
from typing import Dict, List

from openai import OpenAI

"""
Start server with 

python -m vllm.entrypoints.openai.api_server \
  --model TechxGenus/gemma-2b-it-AWQ \
  --quantization awq \
  --dtype float16 \
  --enforce-eager \
  --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port 8000

Then run this client to chat with with the local LLM.

If on same LAN, ssh into the server with ssh nick@192.168.0.150,
then run client_chat.py from within the ssh session.
"""

# Point OpenAI SDK at your tunneled local server
os.environ.setdefault("OPENAI_API_KEY", "dummy")  # server ignores it
client = OpenAI(base_url="http://127.0.0.1:8000/v1")

history: List[Dict[str, str]] = []

while True:
    try:
        user = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if user.lower() in {"exit", "quit"}:
        break
    history.append({"role": "user", "content": user})

    t0 = perf_counter()
    resp = client.chat.completions.create(
        model="TechxGenus/gemma-2b-it-AWQ",
        messages=history,
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
    )
    t1 = perf_counter()
    msg = resp.choices[0].message.content
    print(msg)

    # lean tok/s (generated tokens only, using server's token usage when available)
    gen = (resp.usage and resp.usage.completion_tokens) or 0
    if gen and (t1 - t0) > 0:
        print(f"{gen / (t1 - t0):.2f} tok/s")

    history.append({"role": "assistant", "content": msg})
