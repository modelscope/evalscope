# ❓ 常见问题

这里汇集了 EvalScope 使用过程中常见的各类问题及其解决方案。

```{important}
我们建议您在遇到问题时，首先尝试更新到最新的 `main` 分支代码。许多问题可能已在最新版本中得到修复。
```

## 快速导航

- [❓ 常见问题](#-常见问题)
  - [快速导航](#快速导航)
  - [安装与环境](#安装与环境)
  - [模型评测](#模型评测)
    - [评测配置与参数](#评测配置与参数)
    - [结果异常与问题排查](#结果异常与问题排查)
    - [模型与数据集支持](#模型与数据集支持)
    - [框架使用与扩展](#框架使用与扩展)
  - [性能压测 (perf)](#性能压测-perf)
    - [基本用法与配置](#基本用法与配置)
    - [性能指标与问题排查](#性能指标与问题排查)
  - [引用我们](#引用我们)

## 安装与环境

**Q: 如何通过 Docker 使用 EvalScope？**

**A:** 您可以使用 ModelScope 官方提供的镜像，其中已包含 EvalScope。详情请参考[环境安装文档](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F)。

**Q: `pip install evalscope[all]` 安装时编译失败怎么办？**

**A:** 尝试先单独安装 `pip install python-dotenv`，然后再执行 `pip install evalscope[all]`。

**Q: `evalscope[app]` 与其他库（如 `bfcl-eval`）存在环境冲突？**

**A:** 请尝试单独安装这些库，而不是在一条命令中同时安装。对于 `bfcl-eval`，可以尝试 `2025.6.16` 版本。

**Q: 评测时出现 `trust_remote_code=True` 的警告，如何处理？**

**A:** 这是一个提示性警告，不影响评测流程。EvalScope 框架已默认设置 `trust_remote_code=True`，您可以放心使用。

**Q: 在 Notebook 环境中运行评测报错 `RuntimeError: Cannot run the event loop while another loop is running`？**

**A:** 请将评测代码写入一个 Python 脚本文件（`.py`），然后在终端中执行，避免在 Notebook 中运行。


## 模型评测

### 评测配置与参数

**Q: 如何进行 pass@k 评测或一个样本生成多个答案？**

**A:** `pass@k`指标支持所有聚合方法为`mean`的数据集。具体设置方法：
1. 在`TaskConfig`中设置`repeats=k`
2. 在`dataset_args`中设置
`'<dataset_name>': {'aggregation': 'mean_and_pass_at_k'}`
3. 运行评测时，每个样本会重复`k`次，并计算所有`pass@n, 1<=n<=k`指标。

具体可[参考](https://github.com/modelscope/evalscope/pull/964) 里面的例子。除此之外，还支持`mean_and_vote_at_k`, `mean_and_pass_hat_k`聚合方法。

**Q: 如何移除模型输出中的“思考过程”（如 `<think>...</think>`）？**

**A:** 对于`evalscope >= v1.0 `版本，默认自动处理`<think>...</think>`。如果需要自定义处理，可以使用 `--dataset-args` 参数为特定数据集添加 `filters`。例如，移除 ifeval 数据集评测时 `</think>` 之前的内容：
```shell
--dataset-args '{"ifeval": {"filters": {"remove_until": "</think>"}}}'
```
如果您的模型使用不同的思考标签，如 `<|end_of_thinking|>`，只需替换即可。

**Q: 如何使用本地模型作为裁判模型（Judge Model）？**

**A:** 您可以使用 vLLM 等框架将本地模型部署为一个 API 服务，然后在 `--judge-model-args` 中指定其服务地址。

**Q: 如何为裁判模型设置超时时间？**

**A:** 在 `--judge-model-args` 的 `generation_config` 中设置 `timeout` 参数。

**Q: 如何在评测 API 服务时添加自定义请求头（Header）？**

**A:** 在 `generation_config` 中设置 `extra_headers` 即可。
```python
# 示例
task_config = TaskConfig(
    # ...
    generation_config={'extra_headers': {'Authorization': 'Bearer YOUR_TOKEN'}}
)
```

**Q: 如何设置多卡评测？**

**A:** EvalScope 目前不支持数据并行（Data Parallel）。但您可以通过以下方式实现模型并行：
1.  **使用推理服务**：用 vLLM 等框架启动一个多卡推理服务（例如设置 `--tensor-parallel-size`），然后通过 API 方式进行评测。
2.  **本地加载**：在 `--model-args` 中指定 `device_map=auto`，使模型权重自动分配到多个设备上。

**Q: `stream` 参数报错 `unrecognized arguments: --stream True`？**

**A:** `--stream` 是一个开关参数，直接使用即可，无需附加 `True`。正确用法：`--stream`。

### 结果异常与问题排查

**Q: 评测结果明显异常（如精度极低）如何排查？**

**A:** 请遵循以下步骤：
1.  **检查模型接口**：确认模型服务或本地模型能正常生成回复。
2.  **查看预测文件**：检查 `outputs/<timestamp>/predictions/` 目录下的 JSONL 文件，确认模型输出是否符合预期。
3.  **可视化分析**：使用 `evalscope app` 启动可视化界面，直观地查看和分析评测结果。

**Q: 评测结果不稳定，两次运行结果不一致怎么办？**

**A:** 结果不一致通常由采样随机性导致。可以尝试以下方法固定结果：
1.  在 `generation_config` 中设置 `temperature=0`。
2.  设置固定的 `seed` 参数。

**Q: 评测 `humaneval` 等代码生成任务时，分数远低于预期？**

**A:**
1.  **检查模型类型**：请使用 Instruct 或 Chat 模型，Base 模型可能因不遵循指令而产生复读等问题。
2.  **移除思考过程**：部分模型会在代码前生成思考过程，请参考[这里](#评测配置与参数)的方法进行过滤。
3.  **调整生成长度**：默认最大生成长度可能不足，请在 `generation_config` 中适当调高 `max_tokens`。

**Q: `MATH-500` 数据集评测时，答案提取不准或存在误判？**

**A:** 数学问题的答案格式复杂，规则解析难以覆盖所有情况。建议使用 LLM 作为辅助裁判来提升准确率：
```python
# 在 TaskConfig 中设置
judge_strategy=JudgeStrategy.LLM_RECALL,
judge_model_args={
    'model_id': 'qwen2.5-72b-instruct',
    'api_url': '...',
    'api_key': '...'
}
```
参考文档：[裁判模型参数](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#judge)。

**Q: 评测 `alpaca_eval` 时报错 `Connection error`？**

**A:** `alpaca_eval` 需要指定一个裁判模型（Judge Model）进行打分。默认使用 OpenAI API，如果未配置相关 Key 会导致连接失败。请通过 `--judge-model-args` 指定一个可用的裁判模型。

**Q: 评测中断后，如何从断点处继续？**

**A:** 支持断点续评。使用 `--use-cache` 参数并指定上次评测的输出目录路径，即可重用已完成的模型预测和评估结果。

**Q: `evalscope app` 可视化界面无法访问或图表显示异常？**

**A:**
- **无法访问**：尝试升级 `gradio` 版本到 `4.20.0` 或更高。
- **图表异常**：尝试降级 `plotly` 版本到 `5.15.0`。

**Q: 本地加载模型时报错 `Expected all tensors to be on the same device`？**

**A:** 这通常是显存不足导致的。`device_map='auto'` 可能会将部分权重分配到 CPU。请确保有足够显存，或尝试在更小规模的模型上运行。

### 模型与数据集支持

**Q: 如何评测多模态模型（如 Qwen-VL, Gemma3）？**

**A:** 建议使用 vLLM 等框架将多模态模型部署为 API 服务，然后通过 API 方式进行评测。直接本地加载多模态模型进行评测不支持。


**Q: 使用 API 服务评测 `embeddings` 模型时报错 `dimensions is currently not supported`？**

**A:** 在 `generation_config` 中设置 `'dimensions': None` 或不传递该参数。

**Q: 从本地加载数据集报错（如缺少 `dtype`）？**

**A:** 这是一个已知问题。临时解决方案是：手动删除数据集缓存目录下的 `dataset_infos.json` 文件，然后重试。

**Q: 如何使用自定义数据集进行评测？**

**A:** EvalScope 支持自定义数据集。请参考以下文档：
- **LLM 自定义数据集**：[链接](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html)
- **多模态自定义数据集**：[链接](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/custom.html#id3)

**Q: `rouge` 指标的 `-f`, `-p`, `-r` 分别代表什么？**

**A:** 它们是 ROUGE 指标的三个变体：
- **-r (Recall)**: 召回率，衡量生成文本覆盖了多少参考文本的内容。
- **-p (Precision)**: 精确率，衡量生成文本中有多少内容是准确且相关的。
- **-f (F-measure)**: F1-score，精确率和召回率的调和平均数，是综合性指标。

### 框架使用与扩展

**Q: 如何新增一个自定义的 Benchmark？**

**A:** 您可以继承 `DataAdapter` 并实现其方法，然后通过 `@register_benchmark` 装饰器进行注册。详细步骤请参考[新增 Benchmark 文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)。

**Q: 如何适配非 OpenAI 风格的自定义模型 API？**

**A:** 您可以实现自己的 `Model` 类来对接特定的 API 格式。请参考[自定义模型教程](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_model.html)。

**Q: 如何调试或修改 EvalScope 源码？**

**A:** 请使用源码安装方式。将项目克隆到本地后，在项目根目录执行 `pip install -e .`。这样您对代码的任何修改都会立即生效。

**Q: EvalScope 与 OpenCompass 的评测指标为何无法对齐？**

**A:** 不同评测框架在实现细节（如 Prompt 模板、后处理逻辑、指标计算方式）上存在差异，导致结果难以完全对齐。建议在同一框架内进行模型间的横向对比。

## 性能压测 (perf)

### 基本用法与配置

**Q: `evalscope perf` 的 `--url` 参数应该填什么？**

**A:**
- **通用场景**：对于大多数与 OpenAI API 兼容的服务，请使用 `/v1/chat/completions` 端点。
- **`speed_benchmark` 数据集**：此特定数据集用于测试补全（completion）性能，需配合 `/v1/completions` 端点使用，以避免 Chat Template 处理带来的开销。

**Q: 如何使用本地文件作为压测数据集？**

**A:** 指定 `--dataset-path` 并设置 `--dataset line_by_line`，程序会逐行读取文件内容作为 Prompt。

**Q: 如何压测多模态模型？**

**A:** 目前支持 `flickr8k` 数据集进行多模态压测。设置 `--dataset flickr8k` 即可。

**Q: 压测时如何设置 System Prompt？**

**A:** 在 `model` 参数中以 JSON 字符串形式传入，例如：
```shell
--model '{"model": "my-model", "system_prompt": "You are a helpful assistant."}'
```

### 性能指标与问题排查

**Q: 压测 Ollama 时，并发上不去怎么办？**

**A:** 尝试在执行压测命令前，设置环境变量 `export OLLAMA_NUM_PARALLEL=10` (或其他合适的数值) 来增加 Ollama 的并行处理能力。

**Q: TTFT（首 Token 时间）指标为何与 Latency（总延迟）相同？**

**A:** 要准确测量 TTFT，必须在压测命令中添加 `--stream` 参数以启用流式输出。否则，TTFT 将等于接收到完整响应时的总延迟。

**Q: 压测时 TTFT 指标过高，远超单次请求的响应时间？**

**A:** 当并发数超过服务处理能力时，请求会进入队列等待。TTFT 包含排队时间，因此在高并发下 TTFT 变长是正常现象，反映了服务在高负载下的真实表现。

**Q: 压测结果显示为 `nan`？**

**A:** 请检查输入数据格式是否正确。例如，`openqa` 数据集默认使用 JSONL 文件中的 `question` 字段作为 Prompt。如果字段不匹配或文件格式错误，可能导致无法正确处理请求。

**Q: 压测结果如何可视化？**

**A:** `perf` 子命令的结果不适用于 `evalscope app`。但支持通过 `wandb` 或 `swanlab` 进行可视化。请参考[压测结果可视化指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id6)。

## 引用我们

**Q: 我在我的工作中使用了 EvalScope，该如何引用它？**

**A:** 非常感谢！您可以使用下面的 BibTeX 格式来引用我们的工作：
```bibtex
@misc{evalscope_2024,
    title={{EvalScope}: Evaluation Framework for Large Models},
    author={ModelScope Team},
    year={2024},
    url={https://github.com/modelscope/evalscope}
}
```