"""Clean the Qwen-32B rationale data into data/distill/{train,eval}.jsonl (eval stays held out).

Replies in data/Final_data are "Rationale:\\n...\\n\\nAnswer:\\n..." and become
"<think>\\n{rationale}\\n</think>\\n\\n{answer}". data/training_clean has the same chats with the
original rationales, whose verdicts decide which replies are bad.

A chat is cut before the first reply whose original rationale rejects it or flags a flaw
(answer_flaws.json, labelled by hand), or whose rewritten rationale is cut off, near-empty, or
contains <think> tags or CJK text. Chats with no good reply left are dropped.

    uv run python train/prep_distill.py
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "Final_data"
JUDGED = ROOT / "data" / "training_clean"
# Hand labels, "split:line:turn" -> sentences (not included in this repo).
_flaws = Path(__file__).with_name("answer_flaws.json")
FLAWS = json.load(open(_flaws, encoding="utf-8")) if _flaws.exists() else {}
OUT = ROOT / "data" / "distill"
SEP = "\n\nAnswer:\n"
# Training verdicts the 32B wrote into its rationale ("unsuitable" alone also describes tech).
NEGATIVE = re.compile(r"(?:not suitable|unsuitable)(?: as-is)? (?:for|as) (?:training|strict|a strict|clean training|a training)|"
                      r"(?:marked|makes it|making it) (?:not suitable|unsuitable)\s*\.|"
                      r"suitable_for_training\s*=\s*false|should not be used for training", re.I)
VERDICT = re.compile(r"[^.]*suitable[ _]for[ _]training[^.]*\.\s*", re.I)

CHECKS = {
    "short": lambda r: len(r.split()) < 40,
    "cut_off": lambda r: not r.rstrip().endswith((".", "?", "!", ")", '"', "'", "`", "*")),
    "think_tag": lambda r: "think>" in r,
    "cjk": lambda r: re.search(r"[぀-ヿ一-鿿]", r),
}


def parse(content):
    head, answer = content.split(SEP, 1)
    assert head.startswith("Rationale:\n")
    rationale = head[len("Rationale:\n"):]
    if not NEGATIVE.search(rationale):
        rationale = VERDICT.sub("", rationale)   # drop "The response is suitable for training."
    return rationale.strip(), answer.strip()


def clean(split):
    chats = [json.loads(l)["messages"] for l in open(SRC / f"{split}.jsonl", encoding="utf-8")]
    judged = [json.loads(l)["messages"] for l in open(JUDGED / f"{split}.jsonl", encoding="utf-8")]
    assert len(chats) == len(judged)
    kept, reasons, cut = [], Counter(), 0
    for line, (msgs, orig) in enumerate(zip(chats, judged), 1):
        out, good, turn = [], [], 0
        for m, o in zip(msgs, orig):
            assert m["role"] == o["role"]
            if m["role"] != "assistant":
                assert m["content"] == o["content"], f"{split} line {line}: chat differs from original"
                out.append(m)
                continue
            rat, ans = parse(m["content"])
            old_rat, old_ans = parse(o["content"])
            assert ans == old_ans, f"{split} line {line}: answer changed in the rewrite"
            bad = {k for k, f in CHECKS.items() if f(rat)}
            if NEGATIVE.search(old_rat):
                bad.add("self_reject")
            if "think>" in ans:
                bad.add("think_tag")
            if f"{split}:{line}:{turn}" in FLAWS:
                bad.add("answer_flaw")
            if bad:   # later turns build on this reply, so the chat ends before its user turn
                reasons.update(bad)
                break
            out.append({"role": "assistant", "content": f"<think>\n{rat}\n</think>\n\n{ans}"})
            good = list(out)
            turn += 1
        if not good:
            continue
        cut += len(good) < len(msgs)
        kept.append(good)
    print(f"{split}: kept {len(kept)}/{len(chats)} chats ({cut} cut short before a bad reply); "
          f"bad replies: {dict(reasons)}")
    return kept


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval"):
        kept = clean(split)
        if split == "train":
            random.Random(1).shuffle(kept)
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for m in kept:
                f.write(json.dumps({"messages": m}, ensure_ascii=False) + "\n")
        turns = sum(m["role"] == "assistant" for r in kept for m in r)
        print(f"{split}: {len(kept)} chats, {turns} assistant turns")


if __name__ == "__main__":
    main()
