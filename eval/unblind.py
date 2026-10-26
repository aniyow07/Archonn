"""Unblind the A/B verdicts, check swap consistency, and write judgments.jsonl + summary.json.

    python eval/unblind.py
"""
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "run2" / "judge"


def load(name):
    return [json.loads(l) for l in open(OUT / name, encoding="utf-8")]


def to_model(winner, ft_side):
    if winner == "tie":
        return "tie"
    return "ft" if winner == ft_side else "base"


def main():
    key = json.load(open(OUT / "key.json"))
    skey = json.load(open(OUT / "key_swap.json"))
    rules = {r["idx"]: r for r in load("rules.jsonl")}
    rows = []
    for v in load("verdicts_blind.jsonl"):
        k = key[v["pid"]]
        ft_a = k["ft_side"] == "A"
        rows.append({
            "pid": v["pid"], "idx": k["idx"], "turn": rules[k["idx"]]["turn"],
            "ft_side": k["ft_side"], "winner": to_model(v["winner"], k["ft_side"]),
            "margin": v.get("margin", ""), "reason": v["reason"],
            "tags_ft": v["tags_A"] if ft_a else v["tags_B"],
            "tags_base": v["tags_B"] if ft_a else v["tags_A"],
            "words_ft": rules[k["idx"]]["ft"]["words"], "words_base": rules[k["idx"]]["base"]["words"],
        })
    rows.sort(key=lambda r: r["idx"])
    with open(OUT / "judgments.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    tally = Counter(r["winner"] for r in rows)
    clear = Counter(r["winner"] for r in rows if r["margin"] == "clear")
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn"], Counter())[r["winner"]] += 1
    tags = {m: Counter(t for r in rows for t in r[f"tags_{m}"]) for m in ("ft", "base")}

    main_by_pid = {r["pid"]: r["winner"] for r in rows}
    swap = []
    for v in load("verdicts_swap.jsonl"):
        k = skey[v["pid"]]
        swap.append({"swap_pid": v["pid"], "orig_pid": k["orig"], "idx": k["idx"],
                     "orig": main_by_pid[k["orig"]], "swapped": to_model(v["winner"], k["ft_side"]),
                     "reason": v["reason"]})
    with open(OUT / "swap_consistency.jsonl", "w", encoding="utf-8") as fh:
        for s in swap:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    agree = sum(s["orig"] == s["swapped"] for s in swap)
    flips = sum({s["orig"], s["swapped"]} == {"ft", "base"} for s in swap)

    summary = {
        "n": n, "tally": dict(tally), "rate": {k: round(v / n, 3) for k, v in tally.items()},
        "clear_wins": dict(clear), "by_turn": {t: dict(c) for t, c in sorted(by_turn.items())},
        "tags": {m: dict(c.most_common()) for m, c in tags.items()},
        "swap": {"n": len(swap), "agree": agree, "hard_flips": flips,
                 "disagreements": [s for s in swap if s["orig"] != s["swapped"]]},
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("tally", "rate", "clear_wins", "by_turn", "tags")}, indent=1))
    print("swap", agree, "/", len(swap), "agree; hard flips", flips)
    for s in summary["swap"]["disagreements"]:
        print(" ", s["orig_pid"], s["orig"], "->", s["swapped"])


if __name__ == "__main__":
    main()
