import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_ID = "google/gemma-2-2b-it"

device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)

# Streamed generation to stdout
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("Gemma-2 2B-IT chat (streaming).")

"""
Model card: https://huggingface.co/google/gemma-2-2b-it

Chat template:

messages = 
[    
    {"role": "user", "content": "<user message>"},
    {"role": "assistant", "content": "<assistant message>"},
    ...
]
- No role="system" available for the model.
- Alternating user/assistant is required?!

The instruct model is trained (fine-tuned) on a specific string template that the tokenizer.apply_chat_template method applies. 
When the chat template is converted to the relevant string, it looks as follows:

<bos><start_of_turn>user
hi
<end_of_turn>
<start_of_turn>model
model
Hi! 👋  How can I help you today? 😊
<end_of_turn>
<start_of_turn>user
you can't
<end_of_turn>
<start_of_turn>model
...

where <bos> = beginning-of-sequence token, and the <eos> token stops generation.
<start_of_turn> & <end_of_turn> are special tokens that mark the start and end of chat user (trained).
"""

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
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    reply = full.split(history[-1]["content"])[-1].strip()
    history.append({"role": "assistant", "content": reply})
    print()
