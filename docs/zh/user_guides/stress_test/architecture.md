# 架构

Perf 将六类职责独立拆分：

```text
CLI / Service / Python API
          ↓
Config → SuiteRunner → RunEngine
          ↓
Workload + Scheduler + Protocol + Transport
          ↓
RequestObservation
          ↓
Metrics + ResultStore
          ↓
PerfSuiteResult → Console / HTML / SLA / Service
```

`ResolvedRunSpec` 不可变。每个负载点独占 `RunContext`、队列、transport session、结果存储、取消状态、聚合器和 observer，不存在全局运行状态。

Workload 惰性产生类型化的单轮或多轮 item；scheduler 定义到达和并发语义；protocol adapter 定义 wire schema；HTTP transport 负责 I/O 和时间观测。每次尝试先形成统一 observation，再交给指标和存储层。

测试会约束依赖方向：domain、config、transport、protocol、workload、metrics、results 不允许反向依赖 CLI、Service、SLA 或 reporting。
