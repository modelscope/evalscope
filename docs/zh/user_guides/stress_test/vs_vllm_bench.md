# 与 vLLM Bench 的关系

EvalScope perf 与常见 serving benchmark 使用相同的核心概念：单调时钟、TTFT、TPOT、请求/token 吞吐、closed-loop 并发和 open-loop 到达。

EvalScope 额外提供类型化多负载 suite、多轮 trace、workload 插件、可重放的原始 observation、SLA 搜索、Service 接口和自包含报告。只有模型、endpoint、tokenizer、输入输出长度、stream、到达过程、warmup 与成功策略完全一致时，两个工具的结果才适合直接比较。
