"""Summarize gen_stats_*.jsonl: answers, mean/median/max output tokens and cap hits per task."""
import glob, json, statistics

for p in sorted(glob.glob("gen_stats_*.jsonl")):
    rows = [json.loads(l) for l in open(p)]
    for task in ("gsm8k", "ifeval"):
        t = [r["tokens"] for r in rows if r["task"] == task]
        if t:
            cap = sum(r["finish"] == "length" for r in rows if r["task"] == task)
            print(f"{p[10:-6]:8} {task:6} n={len(t):5} mean={statistics.mean(t):6.0f} "
                  f"median={statistics.median(t):6.0f} max={max(t):5} hit_cap={cap}")
