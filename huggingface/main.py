import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Model name from huggingface
model_name = "google/gemma-2-2b-it"

# Tokenizer used the for model input (same as tokenizer model trained with).
# Maps strings to integer IDs and vice versa. Specific token terminates model output.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# bitsandbytes quantisation config
"""
bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=(
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        else torch.float16
    ),
)
"""

bnb_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# Builds the model from a config.json stored on huggingface.
# model.config contains info on parameters, architecture, etc.
# Moves the model to CPU or GPU depending on availability.
# Handles quantisation if needed...
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    # attn_implementation="flash_attention_2",
    quantization_config=bnb_8bit,
    low_cpu_mem_usage=True,
)

# ---- Model Metadata ----
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

dtype = next(model.parameters()).dtype

print(f"Model: {model_name}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Parameter dtype: {dtype}")

# Rough memory estimate (parameters only)
bytes_per_param = torch.tensor([], dtype=dtype).element_size()
model_memory_gb = total_params * bytes_per_param / (1024**3)

print(f"Approx model memory (GB): {model_memory_gb:.2f}")

# ---- VRAM Usage ----
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem_bytes = torch.cuda.memory_allocated()
    print(f"VRAM allocated after model load: {mem_bytes / (1024**3):.2f} GB")

# Parameters + shapes in the architecture
# for name, param in model.named_parameters():
#    print(name, param.shape)


prompt = "Write an essay about the residents of Finsbury Park."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Warmup
_ = model.generate(**inputs, max_new_tokens=10)

num_new_tokens = 20000

start = time.time()
outputs = model.generate(**inputs, max_new_tokens=num_new_tokens)
end = time.time()

elapsed = end - start
tokens_generated = outputs.shape[-1] - inputs["input_ids"].shape[-1]
tps = tokens_generated / elapsed

print(f"\nGenerated tokens: {tokens_generated}")
print(f"Time taken: {elapsed:.2f}s")
print(f"Tokens/sec: {tps:.2f}")

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
