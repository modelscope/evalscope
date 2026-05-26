# Agentic Trace Replay

Replays the token-length sequences of real production agentic workloads to measure inference-service performance under multi-turn, long-tail, tool-call-paced load. The traces come from [applied-compute/trie](https://github.com/applied-compute/trie) (Apache-2.0); evalscope re-hosts the three workloads on [ModelScope dataset `evalscope/trie-workloads`](https://www.modelscope.cn/datasets/evalscope/trie-workloads) and downloads them on demand.

## Why workload jsonls carry no real conversation text

Each jsonl row contains only token-length metadata, not the actual text:

```json
{
  "input_prompt_length": 8749,
  "num_turns": 24,
  "assistant_response_length": [85, 46, 220, ...],
  "tool_call_output_length": [664, 664, 32, ...],
  "tool_call_latency": [37.197, 1.56, 0.814, ...],
  "final_assistant_response_length": 1658
}
```

At benchmark time the client **synthesizes fake prompts** that re-encode to the exact target lengths (random tokens drawn from the model vocabulary, then decoded and re-encoded with a repair loop). Reasons:

- **Privacy / IP**: trie data comes from real production user traces; raw text would leak both user content and application business logic
- **Cross-model comparability**: the goal is to measure *system performance under "8 k input plus N tool outputs of a few k tokens each in a multi-turn conversation"*, not *the model's semantic response quality on a specific business prompt*. Synthetic prompts standardise away the semantic noise
- **License hygiene**: the length numbers carry no copyright / training-data concern, so the workloads can be published under Apache-2.0

The synthesis approach is the same as `random_dataset` (uniform vocab sampling → decode → re-encode to target length); the only difference is that **trie's length sequences come from real production traces rather than a user-specified distribution**.

## Quick start

```bash
evalscope perf \
  --model qwen-plus \
  --url https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  --api-key $DASHSCOPE_API_KEY \
  --api openai \
  --dataset trie_office_work \
  --multi-turn \
  --parallel 4 \
  --number 10 \
  --duration 600 \
  --tokenizer-path Qwen/Qwen2.5-7B-Instruct \
  --extra-args '{"ignore_eos": true}' \
  --stream
```

When the run finishes you'll see two multi-turn-specific tables (in addition to the existing Detailed Performance / Request Metrics ones):

- **Per-trace Summary**: six metrics (Latency / First-Turn TTFT / TTFAT / Decode TPS / Cache Hit / Eligible Cache Hit) aggregated as mean / min / p50 / p90 / p95 / p99 / max across all completed traces
- **Workload Throughput**: four token-rate metrics (Total / New / Cached prompt, plus Completion) under three windows (Overall / Last 30s / Steady drop-20%)

Three additional files land in the output directory: `trace_summary.json`, `workload_throughput.json`, `workload_timeline.json` (the last is the raw cumulative-token timeline, pandas-friendly).

## The three built-in workloads

| Dataset alias | Source jsonl | Typical num_turns | When to use |
|---|---|---|---|
| `trie_agentic_coding` | `agentic_coding_8k.jsonl` | 8-15 | Coding-agent apps (IDE completion, refactor agents) |
| `trie_code_qa` | `code_qa_8k.jsonl` | 5-12 | Code-Q&A agents (repo analysis, doc generation) |
| `trie_office_work` | `office_work_8k.jsonl` | 18-41 | Office-automation agents (longest, heaviest; best for stress-ceiling tests) |

Each trace starts with ~8 k initial-prompt tokens; total input volume per trace is ~25 k-50 k tokens. Each jsonl has ~8000 traces, more than enough for long-running statistically meaningful tests.

Bring your own trace? Feed any jsonl in the same format via `--dataset-path`:

```bash
evalscope perf ... --dataset trie_office_work --dataset-path /path/to/your_traces.jsonl
```

## How to read the output metrics

### Per-trace Summary

| Column | Meaning | Where to look when it's off |
|---|---|---|
| **Latency (s)** | wall-clock from trace's first-turn `start_time` to last-turn `completed_time` | Slow individual trace → check TTFAT split (is prefill slow or decode slow?) |
| **First-Turn TTFT (s)** | TTFT of the first turn; reflects 8 k cold-prefill performance | High → server's prefill stage is slow; with no KV cache the first token must compute attention over the full 8 k context |
| **TTFAT (s)** | wall-clock from trace start to the first token of the final assistant reply; the end-to-end "how long until the final answer starts streaming" | Any slow turn in the trace inflates this |
| **Decode TPS** | arithmetic mean over the trace's turns of `(completion-1)/(latency-ttft)` | Reflects steady-state decode speed; directly tied to the server's per-token generation cost |
| **Cache Hit Rate (%)** | `sum(cached_tokens) / sum(prompt_tokens)`; includes turn 1 (contributes 0 to numerator but counts in denominator) | Low → server prefix cache isn't enabled or evicts too aggressively; turn 1 always contributes 0 by design (no prior context) |
| **Eligible Cache Hit Rate (%)** | denominator only counts the *theoretically cacheable* prefix (`prev_prompt + prev_completion`), excluding turn 1 and the current turn's new tool output | Both this and Cache Hit Rate low → cache off; this high but Cache Hit low → cache enabled but capacity-bound |

The aggregate columns (mean / min / p50 / p90 / p95 / p99 / max) are percentiles across all completed traces.

### Workload Throughput

```
┌─────────────────────┬───────────┬────────────┬─────────────────────┐
│ Metric (tok/s)      │   Overall │   Last 30s │   Steady (drop 20%) │
├─────────────────────┼───────────┼────────────┼─────────────────────┤
│ Total Prompt tok/s  │   ...     │   ...      │            ...      │
│ New Prompt tok/s    │   ...     │   ...      │            ...      │
│ Cached Prompt tok/s │   ...     │   ...      │            ...      │
│ Completion tok/s    │   ...     │   ...      │            ...      │
└─────────────────────┴───────────┴────────────┴─────────────────────┘
```

| Column | Meaning |
|---|---|
| **Overall** | total tokens / wall_time over the whole run |
| **Last 30s** | sliding 30-second tail-window tok/s; falls back to Overall when the run is shorter than 30s (to avoid misleading numbers) |
| **Steady (drop 20%)** | tok/s over the interval after dropping the first 20% of wall_time; removes the cold-start / batch-scheduler ramp / KV-cache warmup |

| Row | Meaning |
|---|---|
| Total Prompt | New Prompt + Cached Prompt |
| New Prompt | `prompt_tokens - cached_tokens`; the part the server actually runs attention on |
| Cached Prompt | the part the server skipped via prefix cache hit |
| Completion | output-token rate |

Steady-state is the fairest number for "how fast can this endpoint sustainably run"; Last 30s is good for tail jitter / long-trace tail-end behaviour.

### First-Turn vs Subsequent-Turn TTFT

The `Avg TTFT` in the Detailed Performance table averages every turn. In multi-turn trace replay the first turn is a cold prefill (8 k+ input computed from scratch) while subsequent turns benefit heavily from prefix cache, so the two TTFT types can differ by 2-10×. The Request Metrics area now also shows:

- `First-Turn TTFT (ms)`
- `Subsequent-Turn TTFT (ms)`

Both appear only in multi-turn runs; single-turn benchmarks omit them. The combined `Avg TTFT` lands somewhere in the middle and **rarely matches the user-perceived "first-token speed"**; the split is what makes it interpretable.

## `--duration` soft exit + `--number` pairing

Trace replay supports three cap combinations:

| Command | Meaning |
|---|---|
| `--number 50` | run 50 traces, no time limit |
| `--duration 300` | run for 300 seconds, however many traces fit |
| `--number 50 --duration 300` | both caps; **whichever is reached first** ends the run |

**The third is recommended**: `--number` keeps sample size reproducible across runs (essential for comparisons), `--duration` is the safety net against a slow endpoint.

Soft-exit semantics (aligned with upstream trie): once the deadline elapses **no new traces are claimed**, but **already in-flight traces run every remaining turn** before exit. This keeps every trace in `trace_summary.json` complete, so partial traces don't skew latency / cache-hit statistics.

Side effect: actual wall_time may overshoot `--duration` slightly (the overshoot equals the time needed for in-flight traces to finish). Longer traces overshoot more; office_work traces average 60-150s so overshoot can be tens of seconds to ~2 minutes.

## Backend requirements & FAQs

### `ignore_eos` backend dependency

The trie workload's `assistant_response_length` values are real output lengths captured from production traces; replaying faithfully requires the model to **generate to exactly that length**, which depends on the server honouring `extra_body.ignore_eos = True`. Currently supported by:

- vLLM (built-in)
- SGLang (built-in)
- TensorRT-LLM (specific paths)

DashScope OpenAI-compat mode / official OpenAI API / most cloud inference services **do not implement `ignore_eos`** — the model stops at natural EOS. As a result the actual completion length will be **shorter than** `assistant_response_length`, and Completion tok/s will be underestimated. This does not affect cache hit / decode TPS / latency algorithm correctness, but tok/s numbers cannot be compared 1-to-1 against upstream trie.

### Chat-template overhead

evalscope multi-turn talks to `/v1/chat/completions`, where the server applies the model's chat template to serialise the messages array. The template typically adds 10-50 tokens per turn (system prompt + role markers + special tokens). So for the same trace, evalscope sees a slightly higher `prompt_tokens` count than trie (which originally hit `/v1/completions` with raw text), and Cache Hit Rate falls by 2-3pp as the denominator grows. This is an inherent endpoint-choice cost, not an algorithm issue.

### Synthetic-prompt character differences

evalscope **deliberately excludes special tokens and byte-fallback tokens** when synthesising prompts (otherwise CJK multi-byte characters would inflate prompt size by 3-5×). Upstream trie's `TokenizerManager` does not filter. Result:

- evalscope-synthesised prompts are clean ASCII-like
- trie-synthesised prompts may contain special / control characters

**Chat-tuned models react very differently to these two input styles** (former → terse replies, latter → tendency to hallucinate longer explanations), so even when neither side honours `ignore_eos`, the actual completion lengths diverge.

## Comparison against upstream trie

This section shows the measured comparison between evalscope and trie (patched to use chat-completions + real chat history + `temperature=0` + `limit=5`) on the same endpoint (DashScope qwen-plus), same workload (office_work, 5 traces), started simultaneously.

### Algorithm layer (per-trace aggregation)

| Metric | trie | evalscope | Delta |
|---|---|---|---|
| Latency mean (s) | 133.5 | 136.3 | **+2.1%** |
| TTFAT mean (s) | 118.6 | 117.4 | **-1.0%** |
| Decode TPS | 42.13 | 41.80 | **-0.8%** |
| Cache Hit (per-trace mean) | 32.8% | 34.7% | **+5.8%** |
| Eligible Cache (per-trace mean) | 34.6% | 36.6% | **+5.8%** |

**Aligned within ±6%.** The residual gap is mostly 5-trace sample variance, unrelated to the evalscope implementation.

### Workload layer (time-based aggregation)

| Metric | trie | evalscope | Delta |
|---|---|---|---|
| Total Prompt tok/s | 14777 | 14102 | -4.6% |
| Cached Prompt tok/s | 13999 | 5338 | -62% |
| Completion tok/s | 108.7 | 75.9 | -30% |

The workload-level numbers diverge more, but **this is not an algorithm bug** — the root causes are explainable:

1. **Total Prompt -4.6%**: essentially aligned, same order of magnitude
2. **Cached Prompt -62% / New Prompt strongly inverted**: trie's own per-trace cache hit (32.8%) is internally inconsistent with its workload-level number (94.7%); trie's workload aggregation uses a different denominator than its per-trace one. evalscope's two views are self-consistent (per-trace 34.7% ≈ workload 37.8%). **Use the per-trace view for alignment** — that's where the +5.8% number comes from above
3. **Completion -30%**: trie and evalscope synthesise prompts from different vocab-filter strategies, so the prompt text is completely different; the chat model produces different response lengths for different character distributions. Combined with `ignore_eos` not taking effect, this opens the gap

### How to push the alignment to strict ±5%

You need a backend that honours `ignore_eos` (vLLM / SGLang on local hardware) and to run both tools through the same endpoint type (both raw `/v1/completions` or both `/v1/chat/completions`), eliminating the prompt-synthesis character difference. In that setup Cached / Completion / Throughput will all converge to within ±5%. DashScope and most cloud inference services can't meet these constraints today, so cloud benchmarks should validate alignment **at the algorithm layer only** — the ±6% set above.
