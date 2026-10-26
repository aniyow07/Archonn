"""Merge a PEFT LoRA adapter into a 16-bit base, one safetensors shard at a time to keep RAM low."""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

base, adapter, out = (Path(p) for p in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)

cfg = json.loads((adapter / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / cfg["r"]
lora = load_file(adapter / "adapter_model.safetensors")

# "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" -> "model.layers.0.self_attn.q_proj.weight"
pairs = {}
for k in lora:
    if ".lora_A." in k:
        target = k.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
        pairs[target] = (lora[k], lora[k.replace(".lora_A.", ".lora_B.")])
print(f"{len(pairs)} LoRA pairs, scale {scale}")

merged = 0
for shard in sorted(base.glob("*.safetensors")):
    tensors = load_file(shard)
    for name, w in tensors.items():
        if name in pairs:
            a, b = pairs[name]
            tensors[name] = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
            merged += 1
    save_file(tensors, out / shard.name, metadata={"format": "pt"})
    print(f"{shard.name}: done")
    del tensors

assert merged == len(pairs), f"merged {merged} of {len(pairs)}"
for f in base.iterdir():
    if f.suffix != ".safetensors" and f.is_file():
        shutil.copy(f, out / f.name)
# Tokenizer and chat template exactly as used in training
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    shutil.copy(adapter / f, out / f)
print(f"merged {merged} weights into {out}")
