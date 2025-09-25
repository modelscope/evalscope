# Qwen3-Omni 模型评测最佳实践

Qwen3-Omni是新一代原生全模态大模型，能够无缝处理文本、图像、音频和视频等多种输入形式，并通过实时流式响应同时生成文本与自然语音输出。在这篇最佳实践中，我们将使用EvalScope框架以[Qwen3-Omni-30B-A3B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct)模型为例进行评测，覆盖模型服务推理性能评测和模型能力评测。

## 安装依赖

首先，安装[EvalScope](https://github.com/modelscope/evalscope)模型评估框架：

```bash
pip install 'evalscope[app,perf]' -U
```