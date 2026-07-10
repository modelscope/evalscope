# 多轮对话性能压测

Conversation 模式会并发运行多个会话，同时保证每个会话内部的 turn 串行依赖。模型的真实回复会追加到上下文，再构造下一轮请求。

```bash
evalscope perf --model qwen-plus --base-url https://example.com/v1 \
  --workload custom_multi_turn --workload-path ./conversations.jsonl --data-source local \
  --mode conversation --concurrency 8 --conversations 100 \
  --workload-options '{"max_turns":8}'
```

JSONL 每行是一个 OpenAI 消息数组。数据中的 assistant 消息只用于划分 turn，实际运行时会被模型真实输出替换。

内置多轮 workload 包括 `random_multi_turn`、中英文 ShareGPT、`swe_smith` 和 `trie_*` trace replay。

结果包含请求级指标、trace 时延/turn 数分布、首轮和后续轮 TTFT、可用时的缓存估计以及 workload 吞吐。Conversation 模式不接受 open-loop 负载或 tokenized completion 输入。
