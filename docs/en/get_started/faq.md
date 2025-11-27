# ❓ FAQ

This section compiles common issues and solutions encountered when using EvalScope.

```{important}
We recommend updating to the latest `main` branch code when encountering issues. Many problems may have already been fixed in the latest version.
```

## Quick Navigation

- [❓ FAQ](#-faq)
  - [Quick Navigation](#quick-navigation)
  - [Installation \& Environment](#installation--environment)
  - [Model Evaluation](#model-evaluation)
    - [Evaluation Configuration \& Parameters](#evaluation-configuration--parameters)
    - [Result Anomalies \& Troubleshooting](#result-anomalies--troubleshooting)
    - [Model \& Dataset Support](#model--dataset-support)
    - [Framework Usage \& Extension](#framework-usage--extension)
  - [Performance Testing (perf)](#performance-testing-perf)
    - [Basic Usage \& Configuration](#basic-usage--configuration)
    - [Performance Metrics \& Troubleshooting](#performance-metrics--troubleshooting)
  - [Citing Us](#citing-us)

## Installation & Environment

**Q: How to use EvalScope through Docker?**

**A:** You can use the official ModelScope image which includes EvalScope. For details, please refer to the [Environment Setup Documentation](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F).

**Q: What to do when compilation fails during `pip install evalscope[all]`?**

**A:** Try installing `pip install python-dotenv` separately first, then execute `pip install evalscope[all]`.

**Q: Environment conflicts between `evalscope[app]` and other libraries (like `bfcl-eval`)?**

**A:** Please try installing these libraries separately rather than in a single command. For `bfcl-eval`, try version `2025.6.16`.

**Q: How to handle the `trust_remote_code=True` warning during evaluation?**

**A:** This is an informational warning that doesn't affect the evaluation process. EvalScope framework has `trust_remote_code=True` set by default, so you can use it safely.

**Q: Error in Notebook environment: `RuntimeError: Cannot run the event loop while another loop is running`?**

**A:** Please write the evaluation code in a Python script file (`.py`) and execute it in the terminal, avoiding running it in Notebook.

## Model Evaluation

### Evaluation Configuration & Parameters

**Q: How to conduct pass@k evaluation or generate multiple answers for a single sample?**

**A:** The `pass@k` metric supports all datasets with `mean` as the aggregation method. Specific setup instructions:
1. Set `repeats=k` in `TaskConfig`
2. Set in `dataset_args`:
`'<dataset_name>': {'aggregation': 'mean_and_pass_at_k'}`
1. When running the evaluation, each sample will be repeated `k` times, and all `pass@n, 1<=n<=k` metrics will be calculated.

For details, please [refer to](https://github.com/modelscope/evalscope/pull/964) the examples provided. Additionally, `mean_and_vote_at_k` and `mean_and_pass_hat_k` aggregation methods are also supported.

**Q: How to remove the "thinking process" (such as `<think>...</think>`) from model outputs?**

**A:** For `evalscope >= v1.0` versions, the `<think>...</think>` is automatically handled by default. If you need custom handling, use the `--dataset-args` parameter to add `filters` for specific datasets. For example, to remove content before `</think>` when evaluating ifeval dataset:
```shell
--dataset-args '{"ifeval": {"filters": {"remove_until": "</think>"}}}'
```
If your model uses different thinking tags like `<|end_of_thinking|>`, simply replace it accordingly.

**Q: How to use a local model as a Judge Model?**

**A:** You can deploy the local model as an API service using frameworks like vLLM, then specify its service address in `--judge-model-args`.

**Q: How to set timeout for judge models?**

**A:** Set the `timeout` parameter in the `generation_config` of `--judge-model-args`.

**Q: How to add custom request headers when evaluating API services?**

**A:** Set `extra_headers` in `generation_config`.
```python
# Example
task_config = TaskConfig(
    # ...
    generation_config={'extra_headers': {'Authorization': 'Bearer YOUR_TOKEN'}}
)
```

**Q: How to set up multi-GPU evaluation?**

**A:** EvalScope currently doesn't support Data Parallel. However, you can achieve model parallelism through:
1.  **Using inference services**: Start a multi-GPU inference service with frameworks like vLLM (e.g., set `--tensor-parallel-size`), then evaluate through API.
2.  **Local loading**: Specify `device_map=auto` in `--model-args` to automatically distribute model weights across multiple devices.

**Q: Error with `stream` parameter: `unrecognized arguments: --stream True`?**

**A:** `--stream` is a switch parameter, use it directly without appending `True`. Correct usage: `--stream`.

### Result Anomalies & Troubleshooting

**Q: How to troubleshoot obviously abnormal evaluation results (like extremely low accuracy)?**

**A:** Please follow these steps:
1.  **Check model interface**: Confirm that the model service or local model can generate responses normally.
2.  **Review prediction files**: Check JSONL files in the `outputs/<timestamp>/predictions/` directory to confirm model outputs meet expectations.
3.  **Visualization analysis**: Use `evalscope app` to start the visualization interface for intuitive result viewing and analysis.

**Q: Unstable evaluation results - inconsistent across runs?**

**A:** Inconsistent results are usually caused by sampling randomness. Try these methods to stabilize results:
1.  Set `temperature=0` in `generation_config`.
2.  Set a fixed `seed` parameter.

**Q: When evaluating `humaneval` and other code generation tasks, scores are much lower than expected?**

**A:**
1.  **Check model type**: Use Instruct or Chat models. Base models may have issues like repetition due to not following instructions.
2.  **Remove thinking process**: Some models generate thinking process before code. Please refer to [this method](#evaluation-configuration--parameters) for filtering.
3.  **Adjust generation length**: Default maximum generation length may be insufficient. Please appropriately increase `max_tokens` in `generation_config`.

**Q: Inaccurate answer extraction or misjudgments when evaluating `MATH-500` dataset?**

**A:** Math problem answer formats are complex, and rule-based parsing can't cover all cases. We recommend using LLM as an auxiliary judge to improve accuracy:
```python
# Set in TaskConfig or DataAdapter
judge_strategy=JudgeStrategy.LLM_RECALL,
judge_model_args={
    'model_id': 'qwen2.5-72b-instruct',
    'api_url': '...',
    'api_key': '...'
}
```
Reference documentation: [Judge Model Parameters](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#judge).

**Q: `Connection error` when evaluating `alpaca_eval`?**

**A:** `alpaca_eval` requires specifying a Judge Model for scoring. It uses OpenAI API by default, and connection will fail if related keys aren't configured. Please specify an available judge model through `--judge-model-args`.

**Q: How to continue from checkpoint after evaluation interruption?**

**A:** Checkpoint resumption is supported. Use the `--use-cache` parameter and specify the output directory path from the previous evaluation to reuse completed model predictions and evaluation results.

**Q: `evalscope app` visualization interface is inaccessible or charts display abnormally?**

**A:**
- **Inaccessible**: Try upgrading `gradio` version to `4.20.0` or higher.
- **Chart anomalies**: Try downgrading `plotly` version to `5.15.0`.

**Q: Error when loading models locally: `Expected all tensors to be on the same device`?**

**A:** This is usually caused by insufficient GPU memory. `device_map='auto'` may allocate some weights to CPU. Please ensure sufficient GPU memory or try running on smaller models.

### Model & Dataset Support

**Q: How to evaluate multimodal models (like Qwen-VL, Gemma3)?**

**A:** We recommend deploying multimodal models as API services using frameworks like vLLM, then evaluating through API. Direct local loading of multimodal models for pure text evaluation may not be fully supported.

**Q: Error when evaluating `embeddings` models via API service: `dimensions is currently not supported`?**

**A:** Set `'dimensions': None` in `generation_config` or don't pass this parameter.

**Q: Error when loading datasets locally (like missing `dtype`)?**

**A:** This is a known issue. Temporary solution: manually delete the `dataset_infos.json` file in the dataset cache directory, then retry.

**Q: How to use custom datasets for evaluation?**

**A:** EvalScope supports custom datasets. Please refer to the following documentation:
- **LLM Custom Datasets**: [Link](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html)
- **Multimodal Custom Datasets**: [Link](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/custom.html#id3)

**Q: What do `-f`, `-p`, `-r` represent in `rouge` metrics?**

**A:** They are three variants of the ROUGE metric:
- **-r (Recall)**: Measures how much of the reference text content is covered by generated text.
- **-p (Precision)**: Measures how much content in generated text is accurate and relevant.
- **-f (F-measure)**: F1-score, harmonic mean of precision and recall, serving as a comprehensive metric.

### Framework Usage & Extension

**Q: How to add a custom Benchmark?**

**A:** You can inherit from `DataAdapter` and implement its methods, then register through the `@register_benchmark` decorator. For detailed steps, please refer to [Adding Benchmark Documentation](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html).

**Q: How to adapt custom model APIs that are not OpenAI-style?**

**A:** You can implement your own `Model` class to interface with specific API formats. Please refer to [Custom Model Tutorial](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_model.html).

**Q: How to debug or modify EvalScope source code?**

**A:** Please use source code installation. After cloning the project locally, execute `pip install -e .` in the project root directory. This way, any modifications to the code will take effect immediately.

**Q: Why can't evaluation metrics align between EvalScope and OpenCompass?**

**A:** Different evaluation frameworks have differences in implementation details (like prompt templates, post-processing logic, metric calculation methods), making it difficult to achieve complete alignment. We recommend conducting horizontal comparisons between models within the same framework.

## Performance Testing (perf)

### Basic Usage & Configuration

**Q: What should I fill in for the `--url` parameter in `evalscope perf`?**

**A:**
- **General scenarios**: For most OpenAI API-compatible services, use the `/v1/chat/completions` endpoint.
- **`speed_benchmark` dataset**: This specific dataset is used for testing completion performance and should be used with the `/v1/completions` endpoint to avoid overhead from Chat Template processing.

**Q: How to use local files as stress test datasets?**

**A:** Specify `--dataset-path` and set `--dataset line_by_line`. The program will read file content line by line as prompts.

**Q: How to stress test multimodal models?**

**A:** Currently supports the `flickr8k` dataset for multimodal stress testing. Set `--dataset flickr8k`.

**Q: How to set System Prompt during stress testing?**

**A:** Pass it as a JSON string in the `model` parameter, for example:
```shell
--model '{"model": "my-model", "system_prompt": "You are a helpful assistant."}'
```

### Performance Metrics & Troubleshooting

**Q: When stress testing Ollama, concurrency won't increase - what to do?**

**A:** Try setting the environment variable `export OLLAMA_NUM_PARALLEL=10` (or other appropriate value) before executing the stress test command to increase Ollama's parallel processing capability.

**Q: Why is TTFT (Time To First Token) the same as Latency (Total Latency)?**

**A:** To accurately measure TTFT, you must add the `--stream` parameter in the stress test command to enable streaming output. Otherwise, TTFT will equal the total latency when receiving complete responses.

**Q: TTFT metric is too high during stress testing, far exceeding single request response time?**

**A:** When concurrency exceeds service processing capacity, requests enter a queue to wait. TTFT includes queuing time, so increased TTFT under high concurrency is normal and reflects the service's real performance under high load.

**Q: Stress test results show `nan`?**

**A:** Please check if input data format is correct. For example, the `openqa` dataset uses the `question` field in JSONL files as prompts by default. If fields don't match or file format is incorrect, it may cause inability to process requests properly.

**Q: How to visualize stress test results?**

**A:** Results from the `perf` subcommand are not suitable for `evalscope app`. However, visualization through `wandb` or `swanlab` is supported. Please refer to [Stress Test Result Visualization Guide](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id6).

## Citing Us

**Q: I used EvalScope in my work - how should I cite it?**

**A:** Thank you very much! You can use the following BibTeX format to cite our work:
```bibtex
@misc{evalscope_2024,
    title={{EvalScope}: Evaluation Framework for Large Models},
    author={ModelScope Team},
    year={2024},
    url={https://github.com/modelscope/evalscope}
}
```