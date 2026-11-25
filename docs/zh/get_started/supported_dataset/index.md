# 支持的数据集

本章节列出了EvalScope框架原生支持的和通过其他框架支持的评测数据集。

```{tip}
若您需要的数据集不在列表中，可以提交[issue](https://github.com/modelscope/evalscope/issues)，我们会尽快支持；也可以参考[基准评测添加指南](../../advanced_guides/add_benchmark.md)，自行添加数据集并提交[PR](https://github.com/modelscope/evalscope/pulls)，欢迎贡献。

多模态模型评测推荐使用 Native 后端，已原生支持 OCRBench、MMMU、MMBench、MathVista、ChartQA、DocVQA 等主流评测集，详见 [VLM 评测集](vlm.md)。如有特殊需求，也可使用本框架集成的其他工具进行评测，如 [OpenCompass](../../user_guides/backend/opencompass_backend.md) 进行语言模型评测；或使用 [VLMEvalKit](../../user_guides/backend/vlmevalkit_backend.md) 进行多模态模型评测。
```

## 导出评测集目录

使用 `benchmark-info --list` 查看当前安装版本的评测集目录。通过 `--tag`
按标签筛选（不区分大小写），通过 `--format` 导出：

```bash
evalscope benchmark-info --list --tag Coding --format json > coding-benchmarks.json
evalscope benchmark-info --list --format markdown > benchmarks.md
```

JSON 输出为按名称排序的数组，包含 `name`、`pretty_name`、`category`、
`tags`（数组），以及 `metrics` 和 `subsets` 数量。筛选结果为空时输出 `[]`，
不附加表格或总数，便于脚本直接解析标准输出。默认文本格式仍为可读表格。

:::{toctree}
:maxdepth: 2

llm.md
vlm.md
agent.md
aigc.md
other/index.md
:::
