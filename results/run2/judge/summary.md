# run2 judging: base Qwen3-8B vs fine-tuned (32B-rationale distill)

186 held-out chats, one held-out assistant turn each. Both models ran with thinking on. I judged only the answer after `</think>`.

## Headline

| | Fine-tuned wins | Base wins | Tie |
|---|---|---|---|
| All 186 | **38 (20%)** | **125 (67%)** | 23 (12%) |
| Clear wins only | 9 | 66 | – |

By turn depth: turn 2: 16 / 48 / 12, turn 4: 10 / 44 / 4, turn 6: 12 / 33 / 7 (ft / base / tie). The fine-tune loses at every depth.

**In short:** the fine-tune picked up the target *style* almost perfectly. It writes short, conversational answers, ends with "My pick:" and a question, uses no headers or code, and keeps its thinking under 710 tokens. It did not pick up the *substance*. It states wrong facts and wrong arithmetic far more often than base Qwen3, often confidently and sometimes in contradiction with its own earlier turns. Base Qwen3 is wordier, more generic and more often just agrees with the user, but it is wrong less often. That is why it wins most pairs.

## Rule checks (`rules.py`, `rules_summary.json`)

| Check | Base | Fine-tuned | Reference |
|---|---|---|---|
| `</think>` closed | 186/186 | 186/186 | – |
| Empty reply | 0 | 0 | – |
| Ends with a question | 32 | 137 | 132 |
| Recommendation phrase in last 3 lines (heuristic) | 119 | 167 | 152 |
| Ends on a question with no recommendation | 16 | 18 | 30 |
| Code block when the user never asked for code | 4 | 0 | 0 |
| Markdown headers | 46 | 0 | 0 |
| Over 250 words | 26 | 1 | 4 |
| Median / p90 words | 177 / 280 | 178 / 213 | 188 / 224 |
| "the assistant" in the reply | 0 | 0 | 0 |
| "the assistant" in the think block | 10 | 0 | 0 |
| Median / max generated tokens | 918 / 2652 | 474 / 706 | – |

On the rule checks, the fine-tune matches the reference shape very closely. Base breaks the house style more often: headers, checklists, emoji, occasional code, long answers, and "the assistant" in its thinking.

The "ends with a question" check isn't a failure by itself. The reference also ends with a follow-up question in 132 of 186 cases, usually right after a "My pick:" line. The check that matters is "ends on a question with no recommendation" (16 vs 18), and there the two models are about even.

## Failure patterns

Each pair got a reason, plus tags per side (`judgments.jsonl`). Tag counts across 186 answers:

| Tag | Fine-tuned | Base |
|---|---|---|
| fact (wrong technical claim) | **151** | 90 |
| math (wrong arithmetic or units) | **69** | 31 |
| incoherent (self-contradiction, garbled reasoning) | **67** | 9 |
| contradicts an earlier turn | 11 | 5 |
| accepts a wrong premise / no pushback | 12 | 20 |
| missed the key issue | 36 | 56 |
| format (headers or checklist style) | 0 | **132** |
| vague / generic | few | 43 |
| verbose | ~1 | 30 |

**Fine-tuned model:**
- **Invented specifics.** It makes up flags, APIs and settings: `crdb_internal.uuid_v4_prefix_length`, an `rdsoutput` plugin, `MAX_STREAM_RATE`, a `github.com` Alertmanager receiver, `continuous_batching_fraction`, `cuvidEncodePicture`, an `http2: true` option in node-fetch. It also states wrong hardware facts, like AV1 encode on a 3090, a "200GB" RTX 4090 and a 64GB "4090 Ti".
- **Arithmetic and unit errors.** Examples: 100 KB/s written as 100 MB/s (864 GB/day, idx 22). A 1.35 GB/s JSON stream called 1.5 MB/s (idx 113). A 64-byte hash per entity giving 800 MB/s (idx 142). A 4 TB drive said to "easily" hold 3.6 TB/day (idx 173). Activations computed as tokens × parameters (idx 167, where base made the same mistake).
- **Confident but self-contradictory.** It opens with the house-style verdict ("No, that breaks…" / "That's solid, do it.") and then argues something incompatible. Sometimes it flips advice from its own earlier turn: idx 112 caves on a correct 74-hour estimate, and another answer pushes LWT/CAS after the previous turn said to drop it.
- **Sycophantic flips under pushback.** This happened less often than for base, but it is more damaging when it happens, because the reply sounds authoritative.

**Base Qwen3:**
- **Generic checklists.** Numbered "Pros/Cons/Recommendation" lists that are often correct but don't engage with the numbers, and rarely do the math the system prompt asks for.
- **Agrees too easily.** It says "Your plan is solid / Your math checks out" on flawed premises, for example: 15ms drift being "less than" 5ms (idx 13), 76GB for 50M float32 vectors (idx 72), a 5-second idle timeout (idx 96).
- **Formatting violations.** Headers, bold walls, emoji and occasional unrequested code, against "short and conversational".
- When it is wrong, it is usually wrong in a vaguer way, and that gets punished less.

**Shared blind spots:** both models miss most of the reference's specific key insights. Examples: hash-sharded indexes, ES `gc_deletes`, Unsloth being single-GPU, Argo auto-sync overwriting rollbacks, lockstep being the low-bandwidth option for RTS games, a 70B model being unable to emit 100 tokens in under a second.

## Illustrative examples (eval idx → pair id)

- **Fine-tune wins on pushback, idx 24 (P027):** the user's homemade "lowest IP wins" leader election. The fine-tune explains that a dropped socket doesn't mean the node is dead and points to etcd/Consul. Base mostly endorses the scheme.
- **Fine-tune wins on catching the error, idx 72 (P126):** the user computed 76GB for 50M×1536 vectors. The fine-tune catches the missing ×4 bytes (307GB). Base accepts 76GB and assumes 5–10× zstd compression.
- **Base wins, fine-tune arithmetic, idx 22 (P149):** CDC at 50 commits/s × 2KB. The fine-tune says "100MB/sec … 864GB/day" and then recommends replacing CDC with a "pgvector COPY into Elasticsearch".
- **Base wins, fine-tune invention, idx 47 (P021):** the fine-tune claims a PM9A3 NVMe does "100–200 IOPS per core" and tells the user to drop to a single 10G NIC.
- **Base wins, fine-tune caves, idx 112 (P170):** the previous turn correctly said ~74 hours. Under mild pushback, the fine-tune switches to "1.3 hours per epoch" with invented throughput.
- **Base loses by agreeing, idx 96 (P153):** the user's RunPod plan has a 5s idle timeout and expects zero queue wait. Base says "Your math checks out", but the fine-tune's answer was garbled, so base still won that pair.

## Method and caveats

- **Mapping.** The gen files keep their own `history`/`reference`. Their `idx` is the row index in the VM's `data/eval.jsonl`, which is not the same order as `data/Final_data/eval.jsonl`. I matched every item to Final_data by the first user message: all 186 match, at 186 of 203 rows. Formatting differs slightly (the VM reference uses `<think>`, Final_data uses `Rationale:`), but the content is the same. `history` is identical between the base and fine-tuned files.
- **Blinding.** Pairs were shuffled with seed 2026, A/B sides were randomized, and the key was written to `key.json`, which I did not read until all 186 verdicts were saved. Blinding was only partial: the fine-tune's "My pick:" style and base's headers make the source easy to guess, so treat this as identity-hidden in name only.
- **Position bias check.** I re-judged 30 randomly sampled pairs (seed 7) with A/B swapped. All 30 verdicts agreed and none flipped between base and fine-tune (`swap_consistency.jsonl`). This is not an independent check, because I had read those same pairs earlier and very likely remembered them. Read it as "no sign of position bias", not as proof of none.
- **Single judge.** Every verdict is my own reading. Where both answers had serious errors, I picked the less harmful one or called a tie. Tags are coarse and assigned by hand.

## Files

- `rules.py`, `rules.jsonl`, `rules_summary.json`: rule checks per answer (base, fine-tune, reference).
- `packets.py`, `packets/`, `key.json`, `key_swap.json`: the blinded packets and their keys.
- `verdicts_blind.jsonl`, `verdicts_swap.jsonl`: raw A/B verdicts, as written before unblinding.
- `unblind.py`, `judgments.jsonl`: per-item verdicts mapped to base/fine-tune, with reasons, tags and word counts.
- `summary.json`, `swap_consistency.jsonl`: totals and the swap check.
