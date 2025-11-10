import json
import math
import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# --- Model Configuration ---
MODEL_ID = "google/gemma-3-270m-it"  # "google/gemma-2-2b-it"
DEVICE = "cuda"
DTYPE = torch.bfloat16
EPOCHS = 100
LR = 5e-5
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.03
GRAD_CLIP = 1.0
SAVE_DIR = "finetune-gemma-3-270m-it"


# --- Model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
)

# Sets model to training mode, meaning dropout is enabled
# Batch norm continually updates running stats
model.train()


# --- Dataset ---
class TextDataset(Dataset):
    def __init__(
        self, texts: list[str], tokenizer: AutoTokenizer, max_length: int = 64 * 2
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        # Number of samples in the dataset (strings to train on)
        return len(self.texts)

    def __getitem__(self, idx: int):
        # Returns a single sample from the dataset at index idx
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_text(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    return texts


texts: list[str] = load_text("data/tdk_rises.json")
dataset = TextDataset(texts, AutoTokenizer.from_pretrained(MODEL_ID))
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# --- Optimizer & Scheduler ---
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

num_update_steps_per_epoch = math.ceil(len(data_loader))
total_training_steps = EPOCHS * num_update_steps_per_epoch
warmup_steps = int(WARMUP_RATIO * total_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer=optim,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
)

# --- Training Loop ---
global_step = 0
for epoch in range(EPOCHS):
    for batch in data_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()
        scheduler.step()
        optim.zero_grad(set_to_none=True)

        global_step += 1
        if global_step % 10 == 0:
            print(f"step {global_step} | loss {loss.item():.4f}")

    # quick (very rough) eval perplexity
    model.eval()
    eval_loss = 0.0
    eval_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            # accumulate mean loss * tokens predicted (mask != -100)
            tokens = (batch["labels"] != -100).sum().item()
            eval_loss += out.loss.item() * tokens
            eval_tokens += tokens
    ppl = math.exp(eval_loss / max(eval_tokens, 1))
    print(f"epoch {epoch + 1} eval ppl: {ppl:.2f}")
    model.train()

# -------------------------
# 7) save
# -------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"saved to: {SAVE_DIR}")

# --- Tests... ---
model.eval()
prompt_msgs = [
    {
        "role": "user",
        "content": "Write something for me... Whatever is in your most immediate memory.",
    }
]
text = tokenizer.apply_chat_template(
    prompt_msgs, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
gen_ids = model.generate(
    **inputs, max_new_tokens=64, do_sample=False, eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
