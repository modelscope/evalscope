<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>


<p align="center">
  <a href="README_zh.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  中文文档</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documents</a>
<p>

> ⭐ If you like this project, please click the "Star" button at the top right to support us. Your support is our motivation to keep going!

## 📋 Contents
- [Introduction](#-introduction)
- [News](#-news)
- [Installation](#️-installation)
- [Quick Start](#-quick-start)
- [Evaluation Backend](#evaluation-backend)
- [Custom Dataset Evaluation](#️-custom-dataset-evaluation)
- [Model Serving Performance Evaluation](#-model-serving-performance-evaluation)
- [Arena Mode](#-arena-mode)
- [Contribution](#️-contribution)
- [Roadmap](#-roadmap)


## 📝 Introduction

EvalScope is [ModelScope](https://modelscope.cn/)'s official framework for model evaluation and benchmarking, designed for diverse assessment needs. It supports various model types including large language models, multimodal, embedding, reranker, and CLIP models.

The framework accommodates multiple evaluation scenarios such as end-to-end RAG evaluation, arena mode, and inference performance testing. It features built-in benchmarks and metrics like MMLU, CMMLU, C-Eval, and GSM8K. Seamlessly integrated with the [ms-swift](https://github.com/modelscope/ms-swift) training framework, EvalScope enables one-click evaluations, offering comprehensive support for model training and assessment 🚀

<p align="center">
  <img src="docs/en/_static/images/evalscope_framework.png" width="70%">
  <br>EvalScope Framework.
</p>

<details><summary>Framework Description</summary>

The architecture includes the following modules:
1. **Model Adapter**: The model adapter is used to convert the outputs of specific models into the format required by the framework, supporting both API call models and locally run models.
2. **Data Adapter**: The data adapter is responsible for converting and processing input data to meet various evaluation needs and formats.
3. **Evaluation Backend**:
    - **Native**: EvalScope’s own **default evaluation framework**, supporting various evaluation modes, including single model evaluation, arena mode, baseline model comparison mode, etc.
    - **OpenCompass**: Supports [OpenCompass](https://github.com/open-compass/opencompass) as the evaluation backend, providing advanced encapsulation and task simplification, allowing you to submit tasks for evaluation more easily.
    - **VLMEvalKit**: Supports [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) as the evaluation backend, enabling easy initiation of multi-modal evaluation tasks, supporting various multi-modal models and datasets.
    - **RAGEval**: Supports RAG evaluation, supporting independent evaluation of embedding models and rerankers using [MTEB/CMTEB](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html), as well as end-to-end evaluation using [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html).
    - **ThirdParty**: Other third-party evaluation tasks, such as ToolBench.
4. **Performance Evaluator**: Model performance evaluation, responsible for measuring model inference service performance, including performance testing, stress testing, performance report generation, and visualization.
5. **Evaluation Report**: The final generated evaluation report summarizes the model's performance, which can be used for decision-making and further model optimization.
6. **Visualization**: Visualization results help users intuitively understand evaluation results, facilitating analysis and comparison of different model performances.

</details>

## ☎ User Groups

Please scan the QR code below to join our community groups:

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  WeChat Group | DingTalk Group
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.jpg" width="160" height="160">  |  <img src="docs/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">


## 🎉 News
- 🔥🔥 **[2024.12.31]** Support for adding benchmark evaluations, refer to the [📖 Benchmark Evaluation Addition Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html); support for custom mixed dataset evaluations, allowing for more comprehensive model evaluations with less data, refer to the [📖 Mixed Dataset Evaluation Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/collection/index.html).
- 🔥 **[2024.12.13]** Model evaluation optimization: no need to pass the `--template-type` parameter anymore; supports starting evaluation with `evalscope eval --args`. Refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html) for more details.
- 🔥 **[2024.11.26]** The model inference service performance evaluator has been completely refactored: it now supports local inference service startup and Speed Benchmark; asynchronous call error handling has been optimized. For more details, refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).
- 🔥 **[2024.10.31]** The best practice for evaluating Multimodal-RAG has been updated, please check the [📖 Blog](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag) for more details.
- 🔥 **[2024.10.23]** Supports multimodal RAG evaluation, including the assessment of image-text retrieval using [CLIP_Benchmark](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/clip_benchmark.html), and extends [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html) to support end-to-end multimodal metrics evaluation.
- 🔥 **[2024.10.8]** Support for RAG evaluation, including independent evaluation of embedding models and rerankers using [MTEB/CMTEB](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html), as well as end-to-end evaluation using [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html).

<details><summary>More</summary>

- 🔥 **[2024.09.18]** Our documentation has been updated to include a blog module, featuring some technical research and discussions related to evaluations. We invite you to [📖 read it](https://evalscope.readthedocs.io/en/refact_readme/blog/index.html).
- 🔥 **[2024.09.12]** Support for LongWriter evaluation, which supports 10,000+ word generation. You can use the benchmark [LongBench-Write](evalscope/third_party/longbench_write/README.md) to measure the long output quality as well as the output length.
- 🔥 **[2024.08.30]** Support for custom dataset evaluations, including text datasets and multimodal image-text datasets.
- 🔥 **[2024.08.20]** Updated the official documentation, including getting started guides, best practices, and FAQs. Feel free to [📖read it here](https://evalscope.readthedocs.io/en/latest/)!
- 🔥 **[2024.08.09]** Simplified the installation process, allowing for pypi installation of vlmeval dependencies; optimized the multimodal model evaluation experience, achieving up to 10x acceleration based on the OpenAI API evaluation chain.
- 🔥 **[2024.07.31]** Important change: The package name `llmuses` has been changed to `evalscope`. Please update your code accordingly.
- 🔥 **[2024.07.26]** Support for **VLMEvalKit** as a third-party evaluation framework to initiate multimodal model evaluation tasks.
- 🔥 **[2024.06.29]** Support for **OpenCompass** as a third-party evaluation framework, which we have encapsulated at a higher level, supporting pip installation and simplifying evaluation task configuration.
- 🔥 **[2024.06.13]** EvalScope seamlessly integrates with the fine-tuning framework SWIFT, providing full-chain support from LLM training to evaluation.
- 🔥 **[2024.06.13]** Integrated the Agent evaluation dataset ToolBench.

</details>

## 🛠️ Installation
### Method 1: Install Using pip
We recommend using conda to manage your environment and installing dependencies with pip:

1. Create a conda environment (optional)
   ```shell
   # It is recommended to use Python 3.10
   conda create -n evalscope python=3.10
   # Activate the conda environment
   conda activate evalscope
   ```

2. Install dependencies using pip
   ```shell
   pip install evalscope                # Install Native backend (default)
   # Additional options
   pip install evalscope[opencompass]   # Install OpenCompass backend
   pip install evalscope[vlmeval]       # Install VLMEvalKit backend
   pip install evalscope[rag]           # Install RAGEval backend
   pip install evalscope[perf]          # Install Perf dependencies
   pip install evalscope[all]           # Install all backends (Native, OpenCompass, VLMEvalKit, RAGEval)
   ```

> [!WARNING]
> As the project has been renamed to `evalscope`, for versions `v0.4.3` or earlier, you can install using the following command:
> ```shell
> pip install llmuses<=0.4.3
> ```
> To import relevant dependencies using `llmuses`:
> ``` python
> from llmuses import ...
> ```

### Method 2: Install from Source
1. Download the source code
   ```shell
   git clone https://github.com/modelscope/evalscope.git
   ```

2. Install dependencies
   ```shell
   cd evalscope/
   pip install -e .                  # Install Native backend
   # Additional options
   pip install -e '.[opencompass]'   # Install OpenCompass backend
   pip install -e '.[vlmeval]'       # Install VLMEvalKit backend
   pip install -e '.[rag]'           # Install RAGEval backend
   pip install -e '.[perf]'          # Install Perf dependencies
   pip install -e '.[all]'           # Install all backends (Native, OpenCompass, VLMEvalKit, RAGEval)
   ```


## 🚀 Quick Start

To evaluate a model on specified datasets using default configurations, this framework supports two ways to initiate evaluation tasks: using the command line or using Python code.

### Method 1. Using Command Line

Execute the `eval` command in any directory:
```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### Method 2. Using Python Code

When using Python code for evaluation, you need to submit the evaluation task using the `run_task` function, passing a `TaskConfig` as a parameter. It can also be a Python dictionary, yaml file path, or json file path, for example:

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

<details><summary>More Startup Methods</summary>

**Using `TaskConfig`**

```python
from evalscope.run import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=['gsm8k', 'arc'],
    limit=5
)

run_task(task_cfg=task_cfg)
```

**Using `yaml` file**

`config.yaml`:
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

**Using `json` file**

`config.json`:
```json
{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "datasets": ["gsm8k", "arc"],
    "limit": 5
}
```

```python
from evalscope.run import run_task

run_task(task_cfg="config.json")
```
</details>

### Basic Parameter
- `--model`: Specifies the `model_id` of the model in [ModelScope](https://modelscope.cn/), which can be automatically downloaded, e.g., [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/summary); or use the local path of the model, e.g., `/path/to/model`
- `--datasets`: Dataset names, supports inputting multiple datasets separated by spaces. Datasets will be automatically downloaded from modelscope. For supported datasets, refer to the [Dataset List](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset.html)
- `--limit`: Maximum amount of evaluation data for each dataset. If not specified, it defaults to evaluating all data. Can be used for quick validation

### Output Results
```
+-----------------------+-------------------+-----------------+
| Model                 | ai2_arc           | gsm8k           |
+=======================+===================+=================+
| Qwen2.5-0.5B-Instruct | (ai2_arc/acc) 0.6 | (gsm8k/acc) 0.6 |
+-----------------------+-------------------+-----------------+
```

## ⚙️ Complex Evaluation
For more customized evaluations, such as customizing model parameters or dataset parameters, you can use the following command. The evaluation startup method is the same as simple evaluation. Below shows how to start the evaluation using the `eval` command:

```shell
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --model-args revision=master,precision=torch.float16,device_map=auto \
 --generation-config do_sample=true,temperature=0.5 \
 --dataset-args '{"gsm8k": {"few_shot_num": 0, "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

### Parameter
- `--model-args`: Model loading parameters, separated by commas in `key=value` format. Default parameters:
  - `revision`: Model version, default is `master`
  - `precision`: Model precision, default is `auto`
  - `device_map`: Model device allocation, default is `auto`
- `--generation-config`: Generation parameters, separated by commas in `key=value` format. Default parameters:
  - `do_sample`: Whether to use sampling, default is `false`
  - `max_length`: Maximum length, default is 2048
  - `max_new_tokens`: Maximum length of generation, default is 512
- `--dataset-args`: Configuration parameters for evaluation datasets, passed in `json` format. The key is the dataset name, and the value is the parameters. Note that it needs to correspond one-to-one with the values in the `--datasets` parameter:
  - `few_shot_num`: Number of few-shot examples
  - `few_shot_random`: Whether to randomly sample few-shot data, if not set, defaults to `true`

Reference: [Full Parameter Description](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html)


## Evaluation Backend
EvalScope supports using third-party evaluation frameworks to initiate evaluation tasks, which we call Evaluation Backend. Currently supported Evaluation Backend includes:
- **Native**: EvalScope's own **default evaluation framework**, supporting various evaluation modes including single model evaluation, arena mode, and baseline model comparison mode.
- [OpenCompass](https://github.com/open-compass/opencompass): Initiate OpenCompass evaluation tasks through EvalScope. Lightweight, easy to customize, supports seamless integration with the LLM fine-tuning framework ms-swift. [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/opencompass_backend.html)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): Initiate VLMEvalKit multimodal evaluation tasks through EvalScope. Supports various multimodal models and datasets, and offers seamless integration with the LLM fine-tuning framework ms-swift. [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: Initiate RAG evaluation tasks through EvalScope, supporting independent evaluation of embedding models and rerankers using [MTEB/CMTEB](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html), as well as end-to-end evaluation using [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html): [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/index.html)
- **ThirdParty**: Third-party evaluation tasks, such as [ToolBench](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html) and [LongBench-Write](https://evalscope.readthedocs.io/en/latest/third_party/longwriter.html).


## 📈 Model Serving Performance Evaluation
A stress testing tool focused on large language models, which can be customized to support various dataset formats and different API protocol formats.

Reference: Performance Testing [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html)

**Supports wandb for recording results**

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)

**Supports Speed Benchmark**

It supports speed testing and provides speed benchmarks similar to those found in the [official Qwen](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html) reports:

```text
Speed Benchmark Results:
+---------------+-----------------+----------------+
| Prompt Tokens | Speed(tokens/s) | GPU Memory(GB) |
+---------------+-----------------+----------------+
|       1       |      50.69      |      0.97      |
|     6144      |      51.36      |      1.23      |
|     14336     |      49.93      |      1.59      |
|     30720     |      49.56      |      2.34      |
+---------------+-----------------+----------------+
```

## 🖊️ Custom Dataset Evaluation
EvalScope supports custom dataset evaluation. For detailed information, please refer to the Custom Dataset Evaluation [📖User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/index.html)


## 🏟️ Arena Mode
The Arena mode allows multiple candidate models to be evaluated through pairwise battles, and can choose to use the AI Enhanced Auto-Reviewer (AAR) automatic evaluation process or manual evaluation to obtain the evaluation report.

Refer to: Arena Mode [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html)

## 👷‍♂️ Contribution

EvalScope, as the official evaluation tool of [ModelScope](https://modelscope.cn), is continuously optimizing its benchmark evaluation features! We invite you to refer to the [Contribution Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html) to easily add your own evaluation benchmarks and share your contributions with the community. Let’s work together to support the growth of EvalScope and make our tools even better! Join us now!

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>

## 🔜 Roadmap
- [ ] Support for better evaluation report visualization
- [x] Support for mixed evaluations across multiple datasets
- [x] RAG evaluation
- [x] VLM evaluation
- [x] Agents evaluation
- [x] vLLM
- [ ] Distributed evaluating
- [x] Multi-modal evaluation
- [ ] Benchmarks
  - [ ] GAIA
  - [ ] GPQA
  - [x] MBPP


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)
