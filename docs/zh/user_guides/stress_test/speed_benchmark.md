# Speed Workload

`speed_benchmark` 和 `speed_benchmark_long` 生成固定规模的合成 prompt，用于受控吞吐对比。需要 token 精确的随机 workload 时应显式配置 tokenizer，并在 `GenerationConfig` 中设置输出长度。

生产容量评估建议使用真实 workload 的多个 closed-loop 负载点，或明确的 open-loop 到达率。报告 token/s 时必须同时给出模型、输入输出长度、并发/到达率、stream、warmup 和成功率，否则结果不可比。
