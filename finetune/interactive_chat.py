import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

CKPT_DIR = "finetune-gemma-3-270m-it"

device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    CKPT_DIR,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)

# (optional) some tiny quality-of-life
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
print(f"{CKPT_DIR} (streaming).")

history = []
while True:
    user_msg = input("You: ").strip()
    if user_msg.lower() in {"exit", "quit"}:
        break

    history.append({"role": "user", "content": user_msg})
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.001,
            top_p=1.0,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # only the newly generated tokens (avoid brittle string split)
    new_tokens = gen_ids[0, inputs["input_ids"].shape[-1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": reply})
    print()
