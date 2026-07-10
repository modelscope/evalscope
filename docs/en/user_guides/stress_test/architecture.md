# Architecture

Perf separates six independent concerns:

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

`ResolvedRunSpec` is immutable. Every load point owns a `RunContext`, queue, transport session, result store, cancellation state, aggregators, and observers; no run state is global.

Workloads produce typed single-turn or conversation items lazily. Schedulers own arrival/concurrency semantics. Protocol adapters own wire schemas. The HTTP transport owns I/O and timing. Every attempt becomes a typed observation before metrics or persistence consume it.

This dependency direction is enforced by tests: domain, config, transport, protocol, workload, metric, and result layers may not import CLI, Service, SLA, or reporting code.
