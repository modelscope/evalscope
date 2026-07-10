# 使用示例

## Open-loop 到达模型

```bash
evalscope perf --model qwen-plus --base-url https://example.com/v1 \
  --api-key "$API_KEY" --workload prompt --prompt '你好' \
  --mode open_loop --request-rate 20 --requests 1000 \
  --max-outstanding 256 --overflow-policy record_drop --arrival poisson
```

超过客户端并发上限的到达会记录为 dropped observation；调度器不会等待空位并把 open-loop 偷偷变成 closed-loop。

## 本地 JSONL 请求

```bash
evalscope perf --model qwen-plus --base-url https://example.com/v1 \
  --workload line_by_line --workload-path ./requests.jsonl --data-source local \
  --concurrency 4 --requests 100
```

## 多负载点

CLI 使用多个 `--load` JSON；Python 使用包含多个类型化 Load 对象的 `BenchmarkSuite.loads`。

## 本地模型服务

使用 `--target-kind local_transformers` 或 `--target-kind local_vllm`。EvalScope 负责启动、ready 检查、日志和退出清理。
