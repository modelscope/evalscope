# Qwen3 模型评测最佳实践

Qwen3 是 Qwen 系列最新一代的大语言模型，提供了一系列密集和混合专家（MoE）模型。基于广泛的训练，Qwen3 在推理、指令跟随、代理能力和多语言支持方面取得了突破性的进展，支持思考模式和非思考模式的无缝切换。在这篇最佳实践中，我们将使用EvalScope框架以Qwen3-32B模型为例进行全面评测，覆盖模型服务推理性能评测、模型能力评测、以及模型思考效率的评测。

## 安装依赖

首先，安装[EvalScope](https://github.com/modelscope/evalscope)模型评估框架：

```bash
pip install 'evalscope[app,perf]' -U
```

## 模型推理服务

首先，我们需要通过OpenAI API兼容的推理服务接入模型能力，以进行评测。值得注意的是，EvalScope也支持使用transformers进行模型推理评测，详细信息可参考[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#id2)。

除了将模型部署到支持OpenAI接口的云端服务外，还可以选择在本地使用vLLM、ollama等框架直接启动模型。这些推理框架能够很好地支持并发多个请求，从而加速评测过程。特别是对于R1类模型，其输出通常包含较长的思维链，输出token数量往往超过1万。使用高效的推理框架部署模型可以显著提高推理速度。

```bash
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-32B --gpu-memory-utilization 0.9 --served-model-name Qwen3-32B --trust_remote_code --port 8801
```