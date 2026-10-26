"""Rule checks on base vs fine-tuned eval answers (and the reference, as a yardstick).

    python eval/rules.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "run2"  # put gen_base.jsonl and gen_ft.jsonl here
OUT = RUN / "judge"

REC = re.compile(r"my pick|recommend|i'd (go|pick|use|stick|start|keep|do|take|choose)|go with|my (call|vote|advice|take)"
                 r"|stick with|verdict|bottom line|you should|ship (it|the)|don't|do not", re.I)
CODE_ASK = re.compile(r"\b(code|snippet|show me|write (me|a|the)|implement|pseudo-?code|sql|query|config|yaml|example)\b", re.I)


def answer(text):
    return text.split("</think>", 1)[1].strip() if "</think>" in text else text


def check(raw, reply, history):
    lines = [l.strip() for l in reply.strip().splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    tail = " ".join(lines[-3:])
    users = " ".join(m["content"] for m in history if m["role"] == "user")
    words = len(reply.split())
    return {
        "think_closed": "</think>" in raw,
        "empty_reply": not reply.strip(),
        "ends_with_question": last.rstrip("*_ )").endswith("?"),
        "recommendation_near_end": bool(REC.search(tail)),
        "question_only_ending": last.rstrip("*_ )").endswith("?") and not REC.search(tail),
        "has_code_block": "```" in reply,
        "code_asked": bool(CODE_ASK.search(users)),
        "code_violation": "```" in reply and not CODE_ASK.search(users),
        "the_assistant_in_reply": bool(re.search(r"\bthe assistant\b", reply, re.I)),
        "the_assistant_in_think": bool(re.search(r"\bthe assistant\b", raw.split("</think>")[0], re.I)),
        "words": words,
        "over_250_words": words > 250,
        "headers": bool(re.search(r"^#{1,4} ", reply, re.M)),
    }


def main():
    base = [json.loads(l) for l in open(RUN / "gen_base.jsonl", encoding="utf-8")]
    ft = [json.loads(l) for l in open(RUN / "gen_ft.jsonl", encoding="utf-8")]
    rows, agg = [], {}
    for b, f in zip(base, ft):
        row = {"idx": b["idx"], "turn": b["turn"]}
        for name, raw in (("base", b["raw"]), ("ft", f["raw"]), ("reference", b["reference"])):
            reply = {"base": b["reply"], "ft": f["reply"]}.get(name, answer(raw))
            row[name] = c = check(raw, reply, b["history"])
            row[name]["gen_tokens"] = {"base": b["gen_tokens"], "ft": f["gen_tokens"]}.get(name)
            a = agg.setdefault(name, {})
            for k, v in c.items():
                if isinstance(v, bool):
                    a[k] = a.get(k, 0) + v
            a.setdefault("words_list", []).append(c["words"])
        rows.append(row)
    with open(OUT / "rules.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n = len(rows)
    summary = {}
    for name, a in agg.items():
        w = sorted(a.pop("words_list"))
        summary[name] = {k: f"{v}/{n}" for k, v in a.items()}
        summary[name]["words_median"] = w[n // 2]
        summary[name]["words_p90"] = w[int(n * 0.9)]
    toks = {k: sorted(r[k]["gen_tokens"] for r in rows) for k in ("base", "ft")}
    for k, t in toks.items():
        summary[k]["gen_tokens_median"] = t[n // 2]
        summary[k]["gen_tokens_max"] = t[-1]
    json.dump(summary, open(OUT / "rules_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
