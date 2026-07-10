# Parameters

Perf configuration is grouped by responsibility instead of one global argument object.

| Group | CLI examples | Python model |
| --- | --- | --- |
| Target | `--model`, `--protocol`, `--base-url`, `--api-key`, timeouts | `TargetConfig` |
| Workload | `--workload`, `--workload-path`, `--data-source`, `--prompt` | `WorkloadConfig` |
| Generation | `--max-tokens`, `--temperature`, `--top-p`, `--no-stream` | `GenerationConfig` |
| Load | `--mode`, `--requests`, `--concurrency`, `--request-rate` | `ClosedLoopLoad`, `OpenLoopLoad`, `ConversationLoad` |
| Runtime | `--seed`, `--dataset-workers`, `--queue-size`, `--progress` | `RuntimeConfig` |
| Output | `--output-root`, `--run-id`, `--overwrite` | `OutputConfig` |

Load modes:

- `closed_loop`: requires `concurrency` and a request count or duration.
- `open_loop`: requires `request_rate`, a request count or duration, and an explicit `max_outstanding` bound.
- `conversation`: requires a conversation workload, concurrency, and a conversation count or duration.

Use repeatable `--load '<JSON>'` arguments for a suite. For example:

```bash
--load '{"mode":"closed_loop","concurrency":1,"request_count":100}' \
--load '{"mode":"closed_loop","concurrency":8,"request_count":400}'
```

`--warmup-count` and `--warmup-ratio` are mutually exclusive. Unsupported workload/protocol/load combinations fail before network traffic starts.

## Legacy CLI arguments

Existing `evalscope perf` commands continue to work through a shallow CLI translation layer. Common mappings include `--url` to `--base-url`, `--api` to `--protocol`, `--dataset` to `--workload`, `--number` to `--requests`, `--parallel` to `--concurrency`, and `--rate --open-loop` to `--request-rate --mode open_loop`. Legacy list sweeps are converted into explicit load specifications.

The CLI prints a deprecation warning with the corresponding new arguments. This compatibility applies only to command-line parsing; the removed Python `Arguments` API and legacy result schema are not restored.
