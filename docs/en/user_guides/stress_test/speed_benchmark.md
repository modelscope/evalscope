# Speed Workloads

`speed_benchmark` and `speed_benchmark_long` generate fixed-size synthetic prompts for controlled throughput comparisons. Use an explicit tokenizer for token-accurate random workloads and set generation limits in `GenerationConfig`.

For production capacity testing, prefer a normal workload with several closed-loop load points or an open-loop arrival rate. Report request throughput, output-token throughput, latency percentiles, and the exact workload configuration together; a token/s number without load context is not comparable.
