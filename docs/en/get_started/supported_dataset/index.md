# Supported Benchmarks

EvalScope supports a variety of datasets for evaluating different types of models, including language models, AIGC models, and other models. Below is a list of the supported datasets categorized by their respective model types.


```{tip}
If the dataset you need is not on the list, you may submit an [issue](https://github.com/modelscope/evalscope/issues), and we will support it as soon as possible. Alternatively, you can refer to the [Benchmark Addition Guide](../../advanced_guides/add_benchmark.md) to add datasets by yourself and submit a [PR](https://github.com/modelscope/evalscope/pulls). Contributions are welcome.

For multimodal evaluation, the Native backend is recommended. It already supports 40+ mainstream VLM benchmarks including OCRBench, MMMU, MMBench, MathVista, ChartQA, DocVQA — see [VLM Benchmarks](vlm.md) for the full list. If needed, you can also use other tools integrated with this framework, such as [OpenCompass](../../user_guides/backend/opencompass_backend.md) for language model evaluation or [VLMEvalKit](../../user_guides/backend/vlmevalkit_backend.md) for multimodal model evaluation.
```

:::{toctree}
:maxdepth: 2

llm.md
vlm.md
agent.md
aigc.md
other/index.md
:::
