# Relationship to vLLM Bench

EvalScope perf follows the same core measurement concepts as serving benchmarks: monotonic timestamps, TTFT, TPOT, request/token throughput, closed-loop concurrency, and open-loop arrivals.

EvalScope additionally provides typed multi-load suites, conversation traces, workload plugins, exact persisted observations, SLA search, Service integration, and self-contained reports. Compare tools only when model, endpoint, tokenizer, input/output lengths, streaming mode, arrival process, warmup, and success policy are identical.
