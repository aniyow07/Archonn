# Archon

Archon is an experiment in distilling a "theorycrafting partner" into Qwen3-8B: a model that
cross-checks system design, ML and engineering plans, pushes back when something is wrong,
sanity-checks the numbers and ends with a clear recommendation, in short conversational replies.

Training is QLoRA on a single GPU. This repo holds the training, eval and judging code plus the
results. It does not include the datasets or the adapter weights.

## Results in one paragraph

The fine-tune learned the target *style* but not the *substance*. In a blind pairwise
comparison on 186 held-out chats, base Qwen3-8B won 125 to 38 (23 ties). The fine-tune writes
the house style almost perfectly (short, no headers, "My pick:" and a follow-up question), but it
states wrong facts and wrong arithmetic far more often than the base model. Plain SFT on a
stronger model's text copied the format, not the reasoning.

## How it works

**Data.** The starting point is about 3,950 synthetic multi-turn chats in which a strong
reference model plays the partner: it reviews a user's plan, checks the numbers and gives a
verdict. Each reference answer was then given a first-person rationale written by a Qwen 32B
model, explaining how one would arrive at that answer. The rationale becomes the `<think>`
block, so the student learns to reason before it answers:

```
<think>
{rationale}
</think>

{answer}
```

**Cleaning.** `train/prep_distill.py` cuts each chat before the first reply whose rationale
rejects it, flags a flaw in the answer, is cut off or near-empty, or contains stray tags or CJK
text. Later turns build on earlier replies, so everything after a bad reply is dropped too. That
left **3,366 training chats** (9,134 assistant turns) and 186 held-out eval chats.

**Training.** `train/sft.py` runs QLoRA rank 32 on `unsloth/Qwen3-8B-unsloth-bnb-4bit` with
thinking on. Each assistant turn is its own sample, with loss on that reply only, because
Qwen3's template keeps the think block only on the latest reply. Two epochs on one L4 took
about 6.3 hours, and eval loss went from **1.64 to 1.51** (`results/run2/log_history.json`).
`train/check_data.py` verifies the template and loss masking with the real tokenizer first.

**Eval generation.** `train/gen_vllm.py` uses vLLM with FP8 weights and continuous batching.
Base and fine-tune get the same prompts: one held-out assistant turn per eval chat, picked with
a fixed seed, with Qwen3's thinking-mode sampling.

**Judging.** `eval/` builds blinded A/B packets (shuffled pairs, randomized sides, key kept
aside until all verdicts are in), runs rule checks on both outputs and the reference, and then
unblinds. A 30-pair re-judge with sides swapped found no position flips.

## Results

| | Fine-tuned wins | Base wins | Tie |
|---|---|---|---|
| All 186 pairs | 38 | 125 | 23 |
| Clear wins only | 9 | 66 | – |

The fine-tune matched the reference shape on every rule check: no markdown headers, almost never
over 250 words, and thinking under 710 tokens (base used up to 2,652). But it was tagged for
wrong facts 151 times against 90 for base, wrong arithmetic 69 against 31, and incoherent
reasoning 67 against 9. It also invents specific flags, APIs and hardware specs. Base loses
points for generic checklists, headers and agreeing too easily, but it is wrong less often.

- Full write-up: [`results/run2/judge/summary.md`](results/run2/judge/summary.md)
- Example answers: [`results/run2/examples.md`](results/run2/examples.md)

An earlier first iteration trained on the same kind of chats without thinking traces. Base won
that comparison 113 to 44, which is what led to adding the rationales.

## Next steps

SFT on another model's text transfers style. The next things to try are logit-level knowledge
distillation from a teacher that shares the tokenizer, or on-policy distillation, where the
student generates and a stronger model scores or corrects its own outputs.

## Data format

Training and eval files are chat JSONL, one chat per line. Every assistant reply is one
`<think>` block followed by the answer. The rows below are made up to show the shape; they are
not from the dataset.

```json
{"messages": [
  {"role": "system", "content": "You are a sharp theorycrafting partner for system design, ML and engineering plans. Cross-check ideas, push back when something is wrong, sanity-check the numbers, and end with a clear recommendation. Keep replies short and conversational. No code unless asked."},
  {"role": "user", "content": "I want to log every API request to Postgres: about 2,000 requests/s at 1 KB each. One table, one index on timestamp. Fine?"},
  {"role": "assistant", "content": "<think>\nFirst check the volume: 2,000 x 1 KB is 2 MB/s, about 170 GB a day. Postgres can take the write rate, but the table grows fast and deleting old rows will hurt. Partitioning by day makes retention a cheap DROP. So the plan works with one change.\n</think>\n\nMostly yes, but that's about 170 GB a day, and deleting old rows from one big table will crawl. Partition by day so retention is a DROP TABLE instead of a DELETE.\n\nMy pick: daily partitions from day one. How long do you need to keep the logs?"}
]}
```

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "Planning to fine-tune a 70B model on a single 24 GB GPU with full fine-tuning. Doable?"},
  {"role": "assistant", "content": "<think>\nThe weights alone in bf16 are about 140 GB, before gradients and optimizer state. That is far past 24 GB. The user needs QLoRA on a smaller model or more GPUs.\n</think>\n\nNo. The weights alone are about 140 GB in bf16, before gradients and optimizer state. On 24 GB you're looking at QLoRA on a model around 8B to 14B.\n\nMy pick: QLoRA on an 8B first, and see if it's good enough. What's the task?"}
]}
```

## Layout

| Path | What |
|---|---|
| `train/prep_distill.py` | Clean rationale data into `data/distill/{train,eval}.jsonl` |
| `train/sft.py` | QLoRA fine-tune (Unsloth + transformers Trainer) |
| `train/check_data.py` | Pre-flight check of chat template and loss masking |
| `train/gen_vllm.py` | Eval generation with vLLM (FP8) |
| `train/gen_eval.py` | Eval generation with Unsloth/transformers (slower fallback) |
| `eval/packets.py`, `eval/rules.py`, `eval/unblind.py` | Blinded pairwise judging and rule checks |
| `ollama/` | LoRA merge script and the Modelfile for running the result in Ollama |
| `results/run2/` | Loss history, adapter config, judge summary, example answers |

## Running

Training and eval generation need a CUDA GPU (an L4 with 24 GB was enough). On the GPU machine:

```
pip install unsloth vllm
python train/check_data.py          # run from a folder with data/train.jsonl and data/eval.jsonl
python train/sft.py --data data --out out
python train/gen_vllm.py --out out/gen_base.jsonl --max-new 4096
python train/gen_vllm.py --adapter out/adapter --out out/gen_ft.jsonl --max-new 2048
```

For judging, copy `gen_base.jsonl` and `gen_ft.jsonl` into `results/run2/`, then run
`python eval/rules.py` and `python eval/packets.py`, judge the packets, and run
`python eval/unblind.py`.

## Running in Ollama

The trained adapter is not published. To run your own adapter from `train/sft.py` in Ollama,
merge it into the base model first, since Ollama does not load the LoRA directly here:

```
python ollama/merge_lora.py <Qwen3-8B-bf16-dir> out/adapter merged
python llama.cpp/convert_hf_to_gguf.py merged --outfile archon-think-f16.gguf
llama.cpp/llama-quantize archon-think-f16.gguf archon-think-q4_k_m.gguf Q4_K_M
ollama create archon:think -f ollama/Modelfile
ollama run archon:think
```

The Modelfile sets the training system prompt, a ChatML template that keeps the think block
only on the latest reply, and Qwen3's thinking-mode sampling. The Q4_K_M model runs fully on an
8 GB GPU.

## License

Apache-2.0, see [LICENSE](LICENSE).
