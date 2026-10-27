#!/bin/bash
# Run MMLU, ARC-Challenge, GSM8K and IFEval with lm-evaluation-harness + vLLM.
# Usage: eval/run_benchmarks.sh <model path or HF id> <name>
#   eval/run_benchmarks.sh Qwen/Qwen3-8B base
#   eval/run_benchmarks.sh merged archon        # adapter merged with ollama/merge_lora.py
# Fixed batch sizes so --use_cache saves progress per batch and a rerun resumes after a crash.
set -euo pipefail
M=$1; NAME=$2
ARGS="pretrained=$M,dtype=bfloat16,gpu_memory_utilization=0.94,max_model_len=10240,seed=1234"
GEN="temperature=0.6,top_p=0.95,top_k=20,do_sample=True,max_gen_toks=8192"
export STATS_NAME=$NAME   # read by patch_lm_eval_stats.py, if applied

# Knowledge: scored by answer likelihood, no thinking
lm_eval --model vllm --model_args "$ARGS" --tasks mmlu,arc_challenge \
  --batch_size 8192 --use_cache "cache/${NAME}_ll" --output_path "bench/$NAME/ll"

# Math and instructions: thinking on, graded on the text after </think>.
# FP8 is for the KV cache only (more answers in flight); weights stay bf16.
lm_eval --model vllm \
  --model_args "$ARGS,kv_cache_dtype=fp8,enable_thinking=True,think_end_token=</think>" \
  --tasks gsm8k,ifeval --apply_chat_template --fewshot_as_multiturn --gen_kwargs "$GEN" \
  --batch_size 256 --use_cache "cache/${NAME}_gen" --log_samples --output_path "bench/$NAME/gen"
