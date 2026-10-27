"""Patch lm-eval's vLLM backend to log each generation's raw length, thinking included.

lm-eval strips everything up to </think> before scoring, so thinking length is otherwise lost.
Writes one JSON line per answer to gen_stats_$STATS_NAME.jsonl in the working directory, and turns
on vLLM's progress bar (lm-eval only shows it with --batch_size auto).

    python eval/patch_lm_eval_stats.py   # run once in the environment that has lm_eval installed
"""
import lm_eval.models.vllm_causallms as m

p = m.__file__
s = open(p).read()
if "gen_stats_" in s:
    raise SystemExit("already patched")
anchor = "                generated_text: str = output.outputs[0].text\n"
assert s.count(anchor) == 1, "lm-eval version not supported (tested on 0.4.13)"
s = s.replace(anchor, anchor + """                with open("gen_stats_%s.jsonl" % os.environ.get("STATS_NAME", "run"), "a") as _f:
                    _f.write(json.dumps({"task": "gsm8k" if "Question:" in str(_context) else "ifeval",
                        "tokens": len(output.outputs[0].token_ids), "finish": output.outputs[0].finish_reason,
                        "t": time.time()}) + chr(10))
""")
s = s.replace('use_tqdm=self.batch_size == "auto",', "use_tqdm=True,")
s = s.replace("\n", "\nimport os, json, time\n", 1)
open(p, "w").write(s)
print("patched", p)
