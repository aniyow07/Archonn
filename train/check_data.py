"""Pre-flight check on the VM: chat template and loss masking, with the real tokenizer.

Fails loudly if the model would learn from anything but the last reply of each sample,
if that reply lost its think block, or if earlier replies kept theirs.
"""
import sys

from transformers import AutoTokenizer

from sft import BASE, encode, load

tok = AutoTokenizer.from_pretrained(BASE)
for name in ("train", "eval"):
    samples = load(f"data/{name}.jsonl")
    lens, trained_lens, too_long = [], [], 0
    for msgs in samples:
        r = encode(tok, msgs, 4096)
        if r is None:
            too_long += 1
            continue
        text = tok.apply_chat_template(msgs, tokenize=False)
        whole = tok(text, add_special_tokens=False)["input_ids"]
        assert whole == r["input_ids"], "split tokenization differs from whole-text tokenization"
        trained = tok.decode([t for t, l in zip(r["input_ids"], r["labels"]) if l != -100])
        assert trained == msgs[-1]["content"].strip() + "<|im_end|>", "trained text is not the last reply"
        assert text.count("<think>") == 1 and text.count("</think>") == 1, \
            "think blocks of earlier replies were not stripped"
        lens.append(len(r["input_ids"]))
        trained_lens.append(len(trained))
    print(f"{name}: {len(samples)} samples, {too_long} too long, max {max(lens)} tokens, "
          f"total {sum(lens)}")

msgs = samples[2] if len(samples[2]) > 3 else samples[0]
print("TEMPLATE:\n" + tok.apply_chat_template(msgs, tokenize=False))
print("GEN PROMPT TAIL:", repr(tok.apply_chat_template(
    msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=True)[-80:]))
print("CHECK OK")
sys.exit(0)
