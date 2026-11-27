# 插件

插件是单独安装的 Python 包，用现有的 `register_*` 装饰器注册模型 API、评测集、指标、压测 API 协议或压测数据集。
无需 fork EvalScope，即可在 `evalscope eval`、`evalscope perf` 和 `evalscope service` 中使用。

## 声明插件

在 `evalscope.plugins` 入口点组中指向一个模块，导入该模块时完成注册：

```toml
# 插件包的 pyproject.toml
[project.entry-points."evalscope.plugins"]
company-audit = "company_audit.evalscope_plugin"
```

```python
# company_audit/evalscope_plugin.py
from . import benchmarks, metrics, model_api, perf_api  # 各模块使用 @register_* 装饰器
```

只安装插件包不会改变 EvalScope 的行为，只有按入口点名称启用后才会导入。

## 启用插件

```bash
evalscope eval --plugin company-audit --eval-type company_api --datasets company_bench ...
evalscope perf --plugin company-audit --api company_protocol ...
```

`--plugin` 可以重复使用，也可以一次跟多个名称。Python 中使用 `TaskConfig(plugins=[...])` 或压测的
`Arguments(plugins=[...])`。插件注册的内容仍通过 `--eval-type`、`--datasets`、`--api` 和 `--dataset` 选择。

## 报错

- 启用的名称没有任何已安装包提供：运行中止，并列出已安装的插件。
- 插件导入失败：运行中止，报错包含插件名、包名和版本以及原始异常；失败前已注册的内容会被撤销。
- 插件注册了已存在的名称（内置的或其他插件的）：运行中止，报冲突错误并指出注册表和名称。压测 API 协议和数据集原本会被静默覆盖，现在同样报错。

## 可复现

每次运行都会在 `plugin_packages` 下记录启用的插件及其包名、版本和注册的名称：评测写入
`configs/task_config.yaml`，压测写入 `benchmark_args.json`。日志中也有一行 `Using plugin ...`。

## 服务与 Web 界面

由运维在启动服务时指定插件：

```bash
evalscope service --plugin company-audit
```

插件在启动时加载，加载失败则服务无法启动。请求按内置名称加上已启用插件注册的名称校验，每个任务子进程加载同一组插件；
请求不能启用其他插件。`GET /api/v1/plugins` 返回已启用的插件及其注册内容。

在 Web 界面中，评估表单的“插件模型 API”提供已启用插件的模型 API，压测表单的“插件 API 协议”提供其协议，
插件评测集会出现在数据集提示和评测集列表中。

## 示例

[`examples/plugins/evalscope_example_plugin`](https://github.com/modelscope/evalscope/tree/main/examples/plugins/evalscope_example_plugin)
注册了模型 API、评测集、指标、压测 API 协议和压测数据集，并附带一个本地示例服务，README 中的命令都可以在本机直接运行。
