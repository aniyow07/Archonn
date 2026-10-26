"""Generate one reply per held-out eval chat, for the base model or the fine-tuned adapter.

Each chat is cut before one assistant turn, picked with a fixed seed so both models get the same prompts.

    python gen_eval.py --model unsloth/Qwen3-8B-unsloth-bnb-4bit --out out/gen_base.jsonl
    python gen_eval.py --model out/adapter      --out out/gen_ft.jsonl
    python gen_eval.py --model out/adapter --out out/gen_ft_think.jsonl --think --max-new 4096

With --think, "raw" keeps the full output and "reply" only what follows </think> (empty if it never closed).
"""
import argparse
import json
import os
import random

from unsloth import FastLanguageModel
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--think", action="store_true", help="enable Qwen3 thinking mode")
    ap.add_argument("--prefill", default="", help="with --think: text to start the reasoning with")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N chats")
    args = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=6144, load_in_4bit=True, dtype=None)
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"

    rng = random.Random(0)
    items = []
    with open(args.data, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            msgs = json.loads(line)["messages"]
            turns = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
            j = rng.choice(turns)
            prompt = tok.apply_chat_template(msgs[:j], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=args.think)
            if args.think:
                # Open the think block: the fine-tune's replies start with it, and it keeps
                # base Qwen3 from skipping its reasoning.
                prompt += "<think>\n" + args.prefill
            items.append({"idx": i, "turn": j, "history": msgs[:j],
                          "reference": msgs[j]["content"], "prompt": prompt})

    # Write to .part and rename at the end, so a crash never leaves a file that looks done.
    with open(args.out + ".part", "w", encoding="utf-8") as out:
        for b in range(0, len(items), args.batch):
            chunk = items[b:b + args.batch]
            enc = tok([c["prompt"] for c in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                     # Qwen3's recommended sampling for each mode
                                     temperature=0.6 if args.think else 0.7,
                                     top_p=0.95 if args.think else 0.8, top_k=20)
            for c, g in zip(chunk, gen):
                new = g[enc["input_ids"].shape[1]:]
                c = {k: v for k, v in c.items() if k != "prompt"}
                if args.think:
                    # Keep the think tags (some tokenizers mark them special), drop eos/pad.
                    text = tok.decode(new, skip_special_tokens=False)
                    for t in (tok.eos_token, tok.pad_token, "<|im_end|>", "<|endoftext|>"):
                        if t:
                            text = text.replace(t, "")
                    c["raw"] = "<think>\n" + args.prefill + text
                    c["gen_tokens"] = int((new != tok.pad_token_id).sum())
                    c["reply"] = text.split("</think>", 1)[1].strip() if "</think>" in text else ""
                else:
                    text = tok.decode(new, skip_special_tokens=True)
                    c["reply"] = text.replace("<think>", "").replace("</think>", "").strip()
                out.write(json.dumps(c, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{min(b + args.batch, len(items))}/{len(items)}", flush=True)
    os.replace(args.out + ".part", args.out)


if __name__ == "__main__":
    main()
