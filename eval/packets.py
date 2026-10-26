"""Build blinded A/B judging packets. key.json maps packet id -> which side is fine-tuned.

    python eval/packets.py             # main pass, 10 pairs per file
    python eval/packets.py --swap 40   # re-judge sample with A/B swapped
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "run2"  # put gen_base.jsonl and gen_ft.jsonl here
OUT = RUN / "judge"


def answer(text):
    return text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()


def render(pid, b, a_txt, b_txt):
    parts = [f"################ PAIR {pid} ################"]
    for m in b["history"][1:]:
        parts.append(f"[{m['role'].upper()}]\n{answer(m['content'])}")
    parts.append(f"[REFERENCE ANSWER]\n{answer(b['reference'])}")
    parts.append(f"=== A ===\n{a_txt}\n\n=== B ===\n{b_txt}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap", type=int, default=0)
    args = ap.parse_args()
    base = [json.loads(l) for l in open(RUN / "gen_base.jsonl", encoding="utf-8")]
    ft = [json.loads(l) for l in open(RUN / "gen_ft.jsonl", encoding="utf-8")]
    (OUT / "packets").mkdir(exist_ok=True)
    if not args.swap:
        rng = random.Random(2026)
        order = list(range(len(base)))
        rng.shuffle(order)  # judge in shuffled order, not eval order
        key, texts = {}, []
        for n, i in enumerate(order):
            pid = f"P{n:03d}"
            ft_is_a = rng.random() < 0.5
            key[pid] = {"i": i, "idx": base[i]["idx"], "ft_side": "A" if ft_is_a else "B"}
            a, b = (ft[i]["reply"], base[i]["reply"]) if ft_is_a else (base[i]["reply"], ft[i]["reply"])
            texts.append(render(pid, base[i], a, b))
        json.dump(key, open(OUT / "key.json", "w"), indent=1)
        prefix = "batch"
    else:
        key = json.load(open(OUT / "key.json"))
        rng = random.Random(7)
        pids = rng.sample(sorted(key), args.swap)
        skey, texts = {}, []
        for n, pid in enumerate(pids):
            k = key[pid]
            i, sid = k["i"], f"S{n:03d}"
            ft_is_a = k["ft_side"] == "B"  # swapped
            skey[sid] = {"orig": pid, "i": i, "idx": k["idx"], "ft_side": "A" if ft_is_a else "B"}
            a, b = (ft[i]["reply"], base[i]["reply"]) if ft_is_a else (base[i]["reply"], ft[i]["reply"])
            texts.append(render(sid, base[i], a, b))
        json.dump(skey, open(OUT / "key_swap.json", "w"), indent=1)
        prefix = "swap"
    for s in range(0, len(texts), 10):
        (OUT / "packets" / f"{prefix}_{s // 10:02d}.txt").write_text("\n\n".join(texts[s:s + 10]), encoding="utf-8")
    print(len(texts), "pairs")


if __name__ == "__main__":
    main()
