# Conversation Performance Testing

Conversation mode runs several conversations concurrently while preserving sequential turn dependencies inside each conversation. Real assistant replies are appended to the context before the next user turn.

```bash
evalscope perf --model qwen-plus --base-url https://example.com/v1 \
  --workload custom_multi_turn --workload-path ./conversations.jsonl --data-source local \
  --mode conversation --concurrency 8 --conversations 100 \
  --workload-options '{"max_turns":8}'
```

Each JSONL line is an OpenAI-style message array. Reference assistant messages define turn boundaries and are replaced by real model output.

Built-in conversation workloads include `random_multi_turn`, `share_gpt_zh_multi_turn`, `share_gpt_en_multi_turn`, `swe_smith`, and the `trie_*` replay workloads.

Conversation results include request metrics, per-trace latency/turn distributions, first/subsequent-turn TTFT, cache estimates when available, and workload throughput. Open-loop load and tokenized completion input are not valid conversation configurations.
