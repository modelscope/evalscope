<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  <a href="README_zh.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  中文文档</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documentation</a>
<p>


> ⭐ If you like this project, please click the "Star" button in the upper right corner to support us. Your support is our motivation to move forward!

## 📝 Introduction

EvalScope is a one-stop LLM evaluation framework built by the [ModelScope Community](https://modelscope.cn/). Just one command to start — it supports model capability evaluation, inference performance stress testing, and result visualization.

```bash
pip install evalscope
evalscope eval --model your-model-name --api-url $OPENAI_API_BASE_URL --api-key $OPENAI_API_KEY --eval-type openai_api --datasets gsm8k --limit 5
```

## ✨ Key Features

- **📚 Comprehensive Evaluation Benchmarks**: Built-in multiple industry-recognized evaluation benchmarks including MMLU, C-Eval, GSM8K, and more.
- **🧩 Multi-modal and Multi-domain Support**: Supports evaluation of various model types including Large Language Models (LLM), Vision Language Models (VLM), Embedding, Reranker, AIGC, and more.
- **🚀 Multi-backend Integration**: Seamlessly integrates multiple evaluation backends including OpenCompass, VLMEvalKit, RAGEval to meet different evaluation needs.
- **🤖 Agent Evaluation Mode**: Drives benchmarks (e.g. GSM8K, AIME, SWE-bench Agentic) inside a controlled multi-turn AgentLoop with pluggable strategies, tools and Docker sandbox; full per-sample Agent Trace is recorded and visualizable.
- **⚡ Inference Performance Testing**: Provides powerful model service stress testing tools, supporting multiple performance metrics such as TTFT, TPOT.
- **📊 Interactive Reports**: Provides WebUI visualization interface, supporting multi-dimensional model comparison, report overview and detailed inspection.
- **⚔️ Arena Mode**: Supports multi-model battles (Pairwise Battle), intuitively ranking and evaluating models.
- **🔧 Highly Extensible**: Developers can easily add custom datasets, models and evaluation metrics.

## 📊 Visualization Preview

EvalScope provides an interactive Web Dashboard for multi-dimensional model comparison and in-depth analysis.

<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/dashboard/dashboard_overview.png" alt="Dashboard" style="width: 100%;" />
      <p>Dashboard Overview</p>
    </td>
    <td style="text-align: center;">
      <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/dashboard/compare_score_tab.png" alt="Model Compare" style="width: 100%;" />
      <p>Model Comparison</p>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/dashboard/report_overview_tab.png" alt="Report Overview" style="width: 100%;" />
      <p>Report Overview</p>
    </td>
    <td style="text-align: center;">
      <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/dashboard/report_predictions_tab.png" alt="Report Predictions" style="width: 90%;" />
      <p>Prediction Details</p>
    </td>
  </tr>
</table>

For details, please refer to [📖 Visualizing Evaluation Results](https://evalscope.readthedocs.io/en/latest/get_started/visualization.html).

## 🎉 What's New

- 🔥 **[2026.06.02]** Refactored **RAG evaluation** module: upgraded to MTEB 2.x and RAGAS 0.4.x, with unified Pydantic-based configs. See the [RAGEval guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/index.html).
- 🔥 **[2026.05.27]** Added **Trie agentic trace replay** for perf benchmarking: three new dataset plugins (`trie_agentic_coding` / `trie_code_qa` / `trie_office_work`) replay real multi-turn agent traces with per-turn token caps and tool-call latency simulation. Also introduced a `--duration` wall-clock budget for all benchmark modes and a `Turn` dataclass for per-turn overrides.
- 🔥 **[2026.05.27]** Added **Vendor Verifier benchmarks** (`k2_verifier`, `kimi_verifier`, `minimax_verifier`) for validating whether third-party API deployments faithfully reproduce official model behavior, with a shared `VendorVerifierAdapter` base class.
- 🔥 **[2026.05.26]** Added the [GAIA](https://evalscope.readthedocs.io/en/latest/third_party/gaia.html) agent benchmark (multi-turn ReAct + `bash` in a Docker sandbox, official rule-based scorer) and generic [MCP server](https://evalscope.readthedocs.io/en/latest/user_guides/agent/native.html#mcp-server-tools) support — any `NativeAgentConfig`-driven benchmark can now plug in stdio / HTTP / SSE MCP servers (`fetch`, web search, GitHub, ...) without per-benchmark wiring.
- 🔥 **[2026.05.22]** Introduced the **External Agent Bridge** mode: evaluate off-the-shelf agent CLIs such as Anthropic's [Claude Code](https://github.com/anthropics/claude-code) and OpenAI's [Codex](https://github.com/openai/codex) directly through EvalScope. The bridge transparently forwards each CLI's LLM traffic (Anthropic Messages / OpenAI Chat / OpenAI Responses, including SSE streaming) to your evaluation model, while recording the full trajectory as an `agent_trace`. Bring-your-own-runner via `@register_runner`. See the [External Agent Bridge guide](https://evalscope.readthedocs.io/en/latest/user_guides/agent/bridge.html).
- 🔥 **[2026.05.19]** Added support for [SWE-bench_Pro](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench_pro.html) and [τ³-bench](https://evalscope.readthedocs.io/en/latest/third_party/tau3_bench.html): SWE-bench_Pro is a more challenging multilingual long-horizon software-engineering benchmark from Scale AI (recommended over the original SWE-bench for less data contamination and broader language coverage; per-instance Docker images are pulled directly from DockerHub, no local image build required); τ³-bench is the v1.0.0 release of the tau-bench family, extending τ²-bench with a new `banking_knowledge` retrieval domain (RAG), 75+ task fixes across existing domains, and pluggable retrieval pipelines (BM25 / embeddings / rerankers / sandbox shell).
- 🔥 **[2026.05.15]** Introduced **Agent Evaluation Mode**: any benchmark based on `DefaultDataAdapter` (GSM8K, AIME, IFEval, etc.) can now be driven through a multi-turn AgentLoop with pluggable strategies (`function_calling` / `react` / `swe_bench_*`), tools (`bash` / `python_exec` / `submit`) and `local` / `docker` environments. Per-sample `agent_trace` is recorded and rendered step-by-step in the dashboard's Predictions tab. See the [Agent Evaluation guide](https://evalscope.readthedocs.io/en/latest/user_guides/agent/native.html) for details.
- 🔥 **[2026.05.08]** Partnered with [LightSeek](https://lightseek.org/) to launch [TokenSpeed](https://lightseek.org/blog/lightseek-tokenspeed.html), a speed-of-light LLM inference engine for agentic workloads. EvalScope provides the SWE-smith benchmarking pipeline — using real coding-agent traces to measure per-GPU throughput (TPM) and per-user latency (TPS) — serving as the official benchmark tool for TokenSpeed performance evaluation. Refer to the [SWE-smith usage guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/multi_turn.html#swe-smith) to get started.
- 🔥 **[2026.05.07]** Replaced the Gradio-based WebUI with a new React + Vite web interface for better performance and user experience.
- 🔥 **[2026.04.23]** Added support for recording performance (perf) metrics during evaluation tasks, enabling simultaneous tracking of model accuracy and inference efficiency metrics such as TTFT, TPOT, and throughput in a single evaluation run.
- 🔥 **[2026.04.17]** Added support for multi-turn conversation performance stress testing, enabling load testing of dialogue-based model services with multi-turn context. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html).
- 🔥 **[2026.04.10]** Added support for [TIR-Bench](https://arxiv.org/abs/2511.01833) (Thinking-with-Images Reasoning Benchmark), a multimodal benchmark evaluating agentic visual reasoning capabilities of vision-language models.
- 🔥 **[2026.03.24]** Added support for Agent Skill. Any agent model that supports Skill/Tool calling can use natural language to drive EvalScope for model evaluation, performance benchmarking, and result visualization.
- 🔥 **[2026.03.09]** Added support for evaluation progress tracking and HTML format visualization report generation.

<details><summary>More historical updates</summary>

- 🔥 **[2026.03.02]** Added support for Anthropic Claude API evaluation. Use `--eval-type anthropic_api` to evaluate models via Anthropic API service.
- 🔥 **[2026.02.03]** Comprehensive update to dataset documentation, adding data statistics, data samples, usage instructions and more.
- 🔥 **[2026.01.13]** Added support for Embedding and Rerank model service stress testing.
- 🔥 **[2025.12.26]** Added support for Terminal-Bench-2.0, which evaluates AI Agent performance on 89 real-world multi-step terminal tasks.
- 🔥 **[2025.12.18]** Added support for SLA auto-tuning model API services.
- 🔥 **[2025.12.16]** Added support for audio evaluation benchmarks such as Fleurs, LibriSpeech; added support for multilingual code evaluation benchmarks such as MultiplE, MBPP.
- 🔥 **[2025.12.02]** Added support for custom multimodal VQA evaluation; added support for visualizing model service stress testing in ClearML.
- 🔥 **[2025.11.26]** Added support for OpenAI-MRCR, GSM8K-V, MGSM, MicroVQA, IFBench, SciCode benchmarks.
- 🔥 **[2025.11.18]** Added support for custom Function-Call (tool invocation) datasets to test whether models can timely and correctly call tools.
- 🔥 **[2025.11.14]** Added support for SWE-bench_Verified, SWE-bench_Lite, SWE-bench_Verified_mini code evaluation benchmarks.
- 🔥 **[2025.11.12]** Added `pass@k`, `vote@k`, `pass^k` and other metric aggregation methods; added support for multimodal evaluation benchmarks such as A_OKVQA, CMMU, ScienceQA, V*Bench.
- 🔥 **[2025.11.07]** Added support for τ²-bench, an extended and enhanced version of τ-bench that includes a series of code fixes and adds telecom domain troubleshooting scenarios.
- 🔥 **[2025.10.30]** Added support for BFCL-v4, enabling evaluation of agent capabilities including web search and long-term memory.
- 🔥 **[2025.10.27]** Added support for LogiQA, HaluEval, MathQA, MRI-QA, PIQA, QASC, CommonsenseQA and other evaluation benchmarks. Thanks to @[penguinwang96825](https://github.com/penguinwang96825) for the code implementation.
- 🔥 **[2025.10.26]** Added support for Conll-2003, CrossNER, Copious, GeniaNER, HarveyNER, MIT-Movie-Trivia, MIT-Restaurant, OntoNotes5, WNUT2017 and other Named Entity Recognition evaluation benchmarks. Thanks to @[penguinwang96825](https://github.com/penguinwang96825) for the code implementation.
- 🔥 **[2025.10.21]** Optimized sandbox environment usage in code evaluation, supporting both local and remote operation modes.
- 🔥 **[2025.10.20]** Added support for evaluation benchmarks including PolyMath, SimpleVQA, MathVerse, MathVision, AA-LCR; optimized evalscope perf performance to align with vLLM Bench.
- 🔥 **[2025.10.14]** Added support for OCRBench, OCRBench-v2, DocVQA, InfoVQA, ChartQA, and BLINK multimodal image-text evaluation benchmarks.
- 🔥 **[2025.09.22]** Code evaluation benchmarks (HumanEval, LiveCodeBench) now support running in a sandbox environment.
- 🔥 **[2025.09.19]** Added support for multimodal image-text evaluation benchmarks including RealWorldQA, AI2D, MMStar, MMBench, and OmniBench, as well as pure text evaluation benchmarks such as Multi-IF, HealthBench, and AMC.
- 🔥 **[2025.09.05]** Added support for vision-language multimodal model evaluation tasks, such as MathVista and MMMU.
- 🔥 **[2025.09.04]** Added support for image editing task evaluation, including the [GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench) benchmark.
- 🔥 **[2025.08.22]** Version 1.0 Refactoring. Break changes, please [refer to](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#switching-to-version-v1-0).
- 🔥 **[2025.07.18]** The model stress testing now supports randomly generating image-text data for multimodal model evaluation.
- 🔥 **[2025.07.16]** Support for [τ-bench](https://github.com/sierra-research/tau-bench) has been added.
- 🔥 **[2025.07.14]** Support for "Humanity's Last Exam" ([Humanity's-Last-Exam](https://modelscope.cn/datasets/cais/hle)).
- 🔥 **[2025.07.03]** Refactored Arena Mode.
- 🔥 **[2025.06.28]** Optimized custom dataset evaluation; enhanced LLM judge usage.
- 🔥 **[2025.06.19]** Added support for the [BFCL-v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3) benchmark.
- 🔥 **[2025.06.02]** Added support for the Needle-in-a-Haystack test.
- 🔥 **[2025.05.29]** Added support for two long document evaluation benchmarks: DocMath and FRAMES.
- 🔥 **[2025.05.16]** Model service performance stress testing now supports setting various levels of concurrency.
- 🔥 **[2025.05.13]** Added support for the ToolBench-Static dataset, DROP and Winogrande benchmarks.
- 🔥 **[2025.04.29]** Added Qwen3 Evaluation Best Practices.
- 🔥 **[2025.04.27]** Support for text-to-image evaluation.
- 🔥 **[2025.04.10]** Model service stress testing tool now supports the `/v1/completions` endpoint.
- 🔥 **[2025.04.08]** Support for evaluating embedding model services compatible with the OpenAI API has been added.
- 🔥 **[2025.03.27]** Added support for AlpacaEval and ArenaHard evaluation benchmarks.
- 🔥 **[2025.03.20]** The model inference service stress testing now supports generating prompts of specified length using random values.
- 🔥 **[2025.03.13]** Added support for the LiveCodeBench code evaluation benchmark.
- 🔥 **[2025.03.11]** Added support for the SimpleQA and Chinese SimpleQA evaluation benchmarks.
- 🔥 **[2025.03.07]** Added support for the QwQ-32B model evaluation.
- 🔥 **[2025.03.04]** Added support for the SuperGPQA dataset.
- 🔥 **[2025.03.03]** Added support for evaluating the IQ and EQ of models.
- 🔥 **[2025.02.27]** Added support for evaluating the reasoning efficiency of models.
- 🔥 **[2025.02.25]** Added support for MuSR and ProcessBench benchmarks.
- 🔥 **[2025.02.18]** Supports the AIME25 dataset.
- 🔥 **[2025.02.13]** Added support for evaluating DeepSeek distilled models.
- 🔥 **[2025.01.20]** Support for visualizing evaluation results.
- 🔥 **[2025.01.07]** Native backend: Support for model API evaluation.
- 🔥🔥 **[2024.12.31]** Support for adding benchmark evaluations.
- 🔥 **[2024.12.13]** Model evaluation optimization.
- 🔥 **[2024.11.26]** The model inference service performance evaluator has been completely refactored.
- 🔥 **[2024.10.31]** The best practice for evaluating Multimodal-RAG has been updated.
- 🔥 **[2024.10.23]** Supports multimodal RAG evaluation.
- 🔥 **[2024.10.8]** Support for RAG evaluation.
- 🔥 **[2024.09.18]** Documentation added blog module.
- 🔥 **[2024.09.12]** Support for LongWriter evaluation.
- 🔥 **[2024.08.30]** Support for custom dataset evaluations.
- 🔥 **[2024.08.20]** Updated the official documentation.
- 🔥 **[2024.08.09]** Simplified the installation process.
- 🔥 **[2024.07.31]** Important change: The package name `llmuses` has been changed to `evalscope`.
- 🔥 **[2024.07.26]** Support for **VLMEvalKit** as a third-party evaluation framework.
- 🔥 **[2024.06.29]** Support for **OpenCompass** as a third-party evaluation framework.
- 🔥 **[2024.06.13]** EvalScope integrates with SWIFT; Integrated the Agent evaluation dataset ToolBench.

</details>

## 🚀 Quick Start

### Installation

```shell
pip install evalscope
```

> For detailed installation instructions (source install, extra dependencies, etc.), please refer to the [📖 Installation Guide](https://evalscope.readthedocs.io/en/latest/get_started/installation.html).

### Method 1. Evaluate an Online Model API (Recommended for beginners, no GPU required)

Supports any OpenAI API-compatible model service. Just set `$OPENAI_API_BASE_URL` and `$OPENAI_API_KEY` and you are ready to go:

```bash
evalscope eval \
 --model your-model-name \
 --api-url $OPENAI_API_BASE_URL \
 --api-key $OPENAI_API_KEY \
 --eval-type openai_api \
 --datasets gsm8k arc \
 --limit 5
```

### Method 2. Evaluate a Local Model

Evaluate a local model (auto-downloaded from ModelScope):

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### Method 3. Using Python Code

```python
from evalscope import run_task, TaskConfig

task_cfg = TaskConfig(
    model='your-model-name',
    api_url='https://your-openai-compatible-endpoint/v1',
    api_key='your_api_key',
    eval_type='openai_api',
    datasets=['gsm8k', 'arc'],
    limit=5
)

run_task(task_cfg)
```

<details><summary><b>💡 Tip:</b> <code>run_task</code> also supports dictionaries, YAML or JSON files as configuration.</summary>

**Using Python Dictionary**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct',
    'datasets': ['gsm8k', 'arc'],
    'limit': 5
}
run_task(task_cfg=task_cfg)
```

**Using YAML File** (`config.yaml`)
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
</details>

### Output Results
After evaluation completion, you will see a report in the terminal in the following format:
```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

**Launch the visualization dashboard**:
```bash
pip install 'evalscope[service]'
evalscope service
```
Visit `http://127.0.0.1:9000` to open the visualization interface.

## 📈 Advanced Usage

### Custom Evaluation Parameters

You can fine-tune model loading, inference, and dataset configuration through command line parameters.

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}' \
 --generation-config '{"do_sample":true,"temperature":0.6,"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0, "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

- `--model-args`: Model loading parameters such as `revision`, `precision`, etc.
- `--generation-config`: Model generation parameters such as `temperature`, `max_tokens`, etc.
- `--dataset-args`: Dataset configuration parameters such as `few_shot_num`, etc.

For details, please refer to [📖 Complete Parameter Guide](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).

### ⚔️ Arena Mode

Arena mode evaluates model performance through pairwise battles between models, providing win rates and rankings, perfect for horizontal comparison of multiple models.

```text
# Example evaluation results
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)
```
For details, please refer to [📖 Arena Mode Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html).

### 🖊️ Custom Dataset Evaluation

EvalScope allows you to easily add and evaluate your own datasets. For details, please refer to [📖 Custom Dataset Evaluation Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/index.html).

## ⚡ Inference Performance Evaluation Tool

EvalScope provides a powerful stress testing tool for evaluating the performance of large language model services.

- **Key Metrics**: Supports throughput (Tokens/s), first token latency (TTFT), token generation latency (TPOT), etc.
- **Result Recording**: Supports recording results to `wandb` and `swanlab`.
- **Speed Benchmarks**: Can generate speed benchmark results similar to official reports.

For details, please refer to [📖 Performance Testing Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).

<p align="center">
    <img src="docs/en/user_guides/stress_test/images/multi_perf.png" style="width: 80%;">
</p>

## 🧪 Other Evaluation Backends
EvalScope supports launching evaluation tasks through third-party evaluation frameworks (we call them "backends") to meet diverse evaluation needs.

- **Native**: EvalScope's default evaluation framework with comprehensive functionality.
- **OpenCompass**: Focuses on text-only evaluation. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/opencompass_backend.html)
- **VLMEvalKit**: Focuses on multi-modal evaluation. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: Focuses on RAG evaluation, supporting Embedding and Reranker models. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/index.html)
- **Third-party Evaluation Tools**: Supports evaluation tasks like [ToolBench](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html).

<details><summary>🏛️ Overall Architecture</summary>

<p align="center">
    <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/EvalScope%E6%9E%B6%E6%9E%84%E5%9B%BE.png" style="width: 70%;">
    <br>EvalScope Overall Architecture.
</p>

1.  **Input Layer**
    - **Model Sources**: API models (OpenAI API), Local models (ModelScope)
    - **Datasets**: Standard evaluation benchmarks (MMLU/GSM8k etc.), Custom data (MCQ/QA)

2.  **Core Functions**
    - **Multi-backend Evaluation**: Native backend, OpenCompass, MTEB, VLMEvalKit, RAGAS
    - **Performance Monitoring**: Supports multiple model service APIs and data formats, tracking TTFT/TPOP and other metrics
    - **Tool Extensions**: Integrates Tool-Bench, Needle-in-a-Haystack, etc.

3.  **Output Layer**
    - **Structured Reports**: Supports JSON, Table, Logs
    - **Visualization Platform**: Supports Web Dashboard, Wandb, SwanLab

</details>


## ❤️ Community & Support

Welcome to join our community to communicate with other developers and get help.

[Discord Group](https://discord.gg/xc66bMxc4h)              |  WeChat Group | DingTalk Group
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.png" width="160" height="160">  |  <img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">

## 👷‍♂️ Contributing

We welcome any contributions from the community! If you want to add new evaluation benchmarks, models, or features, please refer to our [Contributing Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html).

Thanks to all developers who have contributed to EvalScope!

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>

## 📚 Citation

If you use EvalScope in your research, please cite our work:
```bibtex
@misc{evalscope_2024,
    title={{EvalScope}: Evaluation Framework for Large Models},
    author={ModelScope Team},
    year={2024},
    url={https://github.com/modelscope/evalscope}
}
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)
