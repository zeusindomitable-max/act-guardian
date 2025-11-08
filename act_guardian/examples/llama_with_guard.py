from transformers import AutoModelForCausalLM, AutoTokenizer
from act_guardian import act_guardian
from ebc_clip import energy_budget_clip  # optional

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

clip_fn = energy_budget_clip(model) if 'ebc_clip' in globals() else None
guard_fn = act_guardian(model, min_act=1e-3, noise_scale=1e-4, verbose=True)

text = "Hello, world!"
inputs = tokenizer(text, return_tensors="pt")

for step in range(10):
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    if clip_fn: clip_fn()
    guard_fn()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step+1} | Loss: {loss.item():.4f}")
