# EvalScope example plugin

A minimal plugin package showing how to extend EvalScope from a separately installed
distribution, without forking EvalScope. It registers, with the existing decorators:

| Kind | Name | Selected with |
| --- | --- | --- |
| Model API | `example_arith_api` | `evalscope eval --eval-type` |
| Benchmark | `example_arith` | `evalscope eval --datasets` |
| Metric | `example_numeric_match` | the benchmark's `metric_list` |
| Perf API protocol | `example_arith` | `evalscope perf --api` |
| Perf dataset | `example_arith_prompts` | `evalscope perf --dataset` |

All of them talk to a toy HTTP endpoint (`evalscope_example_plugin/server.py`) that speaks a
small non-OpenAI protocol: `POST /v1/arith` with `{"model", "prompt", "max_tokens"}` returns
`{"id", "text", "usage": {"input_tokens", "output_tokens"}}`. The model API sends each sample
once and keeps the raw response and request id in the prediction metadata.

## How a plugin is declared

`pyproject.toml` points an entry point in the `evalscope.plugins` group at a module.
Importing the module runs the `@register_*` decorators:

```toml
[project.entry-points."evalscope.plugins"]
example = "evalscope_example_plugin"
```

Installing the package does not change EvalScope's behavior. The plugin is imported only
when it is enabled by its entry point name with `--plugin example` (or `plugins=['example']`
in `TaskConfig` / perf `Arguments`).

## Run it

From the repository root:

```bash
pip install -e .
pip install -e examples/plugins/evalscope_example_plugin

# Terminal 1: the toy endpoint
python -m evalscope_example_plugin.server --port 8901
```

Evaluation with the custom model API, benchmark and metric:

```bash
evalscope eval --plugin example \
  --eval-type example_arith_api --model arith-demo --api-url http://127.0.0.1:8901/v1 \
  --datasets example_arith
```

Load test with the custom protocol and dataset:

```bash
evalscope perf --plugin example \
  --api example_arith --dataset example_arith_prompts --model arith-demo \
  --url http://127.0.0.1:8901/v1/arith --parallel 2 --number 10 --no-test-connection
```

Each run records the plugin's package name and version: `configs/task_config.yaml` and
`benchmark_args.json` contain a `plugin_packages` entry with the distribution, version and
the names the plugin registered.

## From the dashboard

```bash
evalscope service --plugin example
```

Open the dashboard and, in the task workspace:

- **Evaluation**: choose `example_arith_api` under *Plugin model API*, set the API URL to
  `http://127.0.0.1:8901/v1`, and enter `example_arith` as the dataset.
- **Performance**: choose `example_arith` under *Plugin API protocol*, set the URL to
  `http://127.0.0.1:8901/v1/arith`, and enter `example_arith_prompts` as the dataset.

Only plugins enabled with `evalscope service --plugin` are loaded. Requests and task
workers use that same list, and `GET /api/v1/plugins` reports it.

## Python API

```python
from evalscope import TaskConfig, run_task

run_task(
    TaskConfig(
        plugins=['example'],
        eval_type='example_arith_api',
        model='arith-demo',
        api_url='http://127.0.0.1:8901/v1',
        datasets=['example_arith'],
    )
)
```
