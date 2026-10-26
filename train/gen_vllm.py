"""Eval generation with vLLM (FP8, continuous batching): same prompts, sampling and output
format as gen_eval.py --think. The fine-tune adds its LoRA on top of the same base.

    python gen_vllm.py --out out/gen_base.jsonl --max-new 4096
    python gen_vllm.py --adapter out/adapter --out out/gen_ft.jsonl --max-new 2048
"""
import argparse
import json
import os
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

BASE = "Qwen/Qwen3-8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", default="data/eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    # Same turn choice as gen_eval.py (seed 0, one assistant turn per chat).
    rng = random.Random(0)
    items = []
    with open(args.data, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            msgs = json.loads(line)["messages"]
            j = rng.choice([k for k, m in enumerate(msgs) if m["role"] == "assistant"])
            prompt = tok.apply_chat_template(msgs[:j], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True)
            items.append({"idx": i, "turn": j, "history": msgs[:j],
                          "reference": msgs[j]["content"], "prompt": prompt + "<think>\n"})

    # FP8 leaves ~10 GB of KV cache on the L4 (vs ~3 GB): ~4x more answers at once.
    llm = LLM(BASE, dtype="bfloat16", quantization="fp8", max_model_len=8192, gpu_memory_utilization=0.92,
              enable_lora=bool(args.adapter), max_lora_rank=32, seed=0)
    # Keep special tokens so </think> survives even if the tokenizer marks it special.
    params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_new, seed=0,
                            skip_special_tokens=False)
    lora = LoRARequest("ft", 1, os.path.abspath(args.adapter)) if args.adapter else None
    outs = llm.generate([c["prompt"] for c in items], params, lora_request=lora)

    with open(args.out + ".part", "w", encoding="utf-8") as f:
        for c, o in zip(items, outs):
            text = o.outputs[0].text
            for t in ("<|im_end|>", "<|endoftext|>"):
                text = text.replace(t, "")
            c = {k: v for k, v in c.items() if k != "prompt"}
            c["raw"] = "<think>\n" + text
            c["gen_tokens"] = len(o.outputs[0].token_ids)
            c["reply"] = text.split("</think>", 1)[1].strip() if "</think>" in text else ""
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(args.out + ".part", args.out)


if __name__ == "__main__":
    main()
