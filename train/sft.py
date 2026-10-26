"""QLoRA fine-tune of Qwen3-8B on the distillation chats, thinking on (runs on the L4 VM).

Qwen3's template keeps the think block only on the reply after the last user turn, so each
assistant turn is its own sample (the chat up to that reply) with loss on that reply only.

    python sft.py --data data --out out
"""
import argparse
import json
import os
import random

from unsloth import FastLanguageModel  # must be imported before transformers
import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

# Pre-quantized 4-bit weights: a ~6 GB download instead of ~16 GB, same model.
BASE = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
ASSISTANT_START = "<|im_start|>assistant\n"
END = "<|im_end|>"


def load(path):
    """One sample per assistant turn: the chat up to and including that reply."""
    with open(path, encoding="utf-8") as f:
        chats = [json.loads(line)["messages"] for line in f]
    samples = []
    for msgs in chats:
        for j, m in enumerate(msgs):
            if m["role"] == "assistant":
                c = m["content"]
                assert c.startswith("<think>\n") and c.count("<think>") == 1 \
                    and c.count("</think>") == 1, "reply is not one think block plus answer"
                samples.append(msgs[:j + 1])
    return samples


def encode(tok, messages, max_len):
    """Token ids plus labels that are -100 everywhere except the last assistant reply."""
    text = tok.apply_chat_template(messages, tokenize=False)
    head_end = text.rindex(ASSISTANT_START) + len(ASSISTANT_START)
    end = text.index(END, head_end) + len(END)
    prompt = tok(text[:head_end], add_special_tokens=False)["input_ids"]
    reply = tok(text[head_end:end], add_special_tokens=False)["input_ids"]
    tail = tok(text[end:], add_special_tokens=False)["input_ids"]
    ids = prompt + reply + tail
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(prompt) + reply + [-100] * len(tail)}


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            k = n - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad_id] * k)
            out["labels"].append(b["labels"] + [-100] * k)
            out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * k)
        return {key: torch.tensor(v) for key, v in out.items()}
    return fn


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only N chats")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N steps")
    args = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(
        BASE, max_seq_length=args.max_len, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=0)

    sets = {}
    for name in ("train", "eval"):
        chats = load(f"{args.data}/{name}.jsonl")
        if args.limit:
            chats = chats[:args.limit]
        rows = [r for r in (encode(tok, m, args.max_len) for m in chats) if r]
        lens = sorted(len(r["input_ids"]) for r in rows)
        print(f"{name}: {len(rows)}/{len(chats)} samples kept, tokens p50 {lens[len(lens)//2]} "
              f"max {lens[-1]} total {sum(lens)}", flush=True)
        sets[name] = rows
    random.Random(0).shuffle(sets["train"])

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        weight_decay=0.0,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=0.125,
        # Checkpoint every quarter of the run so a crash or the time limit loses little.
        save_strategy="steps",
        save_steps=0.25,
        save_total_limit=2,
        train_sampling_strategy="group_by_length",
        report_to="none",
        remove_unused_columns=False,
        seed=0,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ListDataset(sets["train"]), eval_dataset=ListDataset(sets["eval"]),
        data_collator=collate(tok.pad_token_id))
    resume = get_last_checkpoint(args.out) if os.path.isdir(args.out) else None
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(f"{args.out}/adapter")
    tok.save_pretrained(f"{args.out}/adapter")
    with open(f"{args.out}/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)


if __name__ == "__main__":
    main()
