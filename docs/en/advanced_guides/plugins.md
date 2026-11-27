# Plugins

A plugin is a separately installed Python package that adds model APIs, benchmarks, metrics,
perf API protocols or perf datasets with the existing `register_*` decorators. It works in
`evalscope eval`, `evalscope perf` and `evalscope service` without forking EvalScope.

## Declare a plugin

Point an entry point in the `evalscope.plugins` group at a module. Importing the module must run
the registrations:

```toml
# pyproject.toml of your package
[project.entry-points."evalscope.plugins"]
company-audit = "company_audit.evalscope_plugin"
```

```python
# company_audit/evalscope_plugin.py
from . import benchmarks, metrics, model_api, perf_api  # each module uses @register_* decorators
```

Installing the package does not change EvalScope's behavior: a plugin is imported only when it is
enabled by its entry point name.

## Enable it

```bash
evalscope eval --plugin company-audit --eval-type company_api --datasets company_bench ...
evalscope perf --plugin company-audit --api company_protocol ...
```

`--plugin` can be repeated or followed by several names. In Python, use `TaskConfig(plugins=[...])`
or perf `Arguments(plugins=[...])`. What the plugin registers is still selected with `--eval-type`,
`--datasets`, `--api` and `--dataset`.

## Errors

- An enabled name that no installed package provides stops the run and lists the installed plugins.
- A plugin that fails to import stops the run. The error names the plugin, its package and version,
  and the original exception; anything it registered before failing is removed.
- A plugin that registers a name that is already registered (built-in or from another plugin)
  stops the run with a conflict error naming the registry and the name. This also applies to perf
  API protocols and datasets, which are otherwise overwritten silently.

## Reproducibility

Each run records the enabled plugins with their package name, version and registered names under
`plugin_packages`: in `configs/task_config.yaml` for evaluations and in `benchmark_args.json` for
perf runs. The log also contains a `Using plugin ...` line.

## Service and dashboard

The operator chooses the plugins when starting the service:

```bash
evalscope service --plugin company-audit
```

They are loaded at startup, so a broken plugin prevents the service from starting. Requests are
validated against the built-in names plus those of the enabled plugins, and every task worker loads
the same plugins. A request may not enable other plugins. `GET /api/v1/plugins` lists the enabled
plugins and what they registered.

In the dashboard, the evaluation form offers the enabled plugins' model APIs under
*Plugin model API*, the perf form offers their protocols under *Plugin API protocol*, and plugin
benchmarks appear in the dataset suggestions and the benchmark list.

## Example

[`examples/plugins/evalscope_example_plugin`](https://github.com/modelscope/evalscope/tree/main/examples/plugins/evalscope_example_plugin)
registers a model API, a benchmark, a metric, a perf API protocol and a perf dataset, and ships a
toy endpoint so that every command in its README runs locally.
