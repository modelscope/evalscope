

![](docs/en/_static/images/evalscope_logo.png)

<p align="center">
    English | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope">
</a>
<a href='https://evalscope.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/evalscope-en/badge/?version=latest' alt='Documentation Status' />
</a>
<br>
 <a href="https://evalscope.readthedocs.io/en/latest/">📖 Documents</a>
<p>


## 📋 Table of Contents
- [Introduction](#introduction)
- [News](#News)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation Backend](#evaluation-backend)
- [Custom Dataset Evaluation](#custom-dataset-evaluation)
- [Offline Evaluation](#offline-evaluation)
- [Arena Mode](#arena-mode)
- [Model Serving Performance Evaluation](#Model-Serving-Performance-Evaluation)


## 📝 Introduction

EvalScope is the official model evaluation and performance benchmarking framework launched by the [ModelScope](https://modelscope.cn/) community. It comes with built-in common benchmarks and evaluation metrics, such as MMLU, CMMLU, C-Eval, GSM8K, ARC, HellaSwag, TruthfulQA, MATH, and HumanEval. EvalScope supports various types of model evaluations, including LLMs, multimodal LLMs, embedding models, and reranker models. It is also applicable to multiple evaluation scenarios, such as end-to-end RAG evaluation, arena mode, and model inference performance stress testing. Moreover, with the seamless integration of the ms-swift training framework, evaluations can be initiated with a single click, providing full end-to-end support from model training to evaluation 🚀

<p align="center">
  <img src="docs/en/_static/images/evalscope_framework.png" width="70%">
  <br>EvalScope Framework.
</p>

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


## 🎉 News
- 🔥 **[2024.10.31]** The best practice for evaluating Multimodal-RAG has been updated, please check the [📖 Blog](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag) for more details.
- 🔥 **[2024.10.23]** Supports multimodal RAG evaluation, including the assessment of image-text retrieval using [CLIP_Benchmark](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/clip_benchmark.html), and extends [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html) to support end-to-end multimodal metrics evaluation.
- 🔥 **[2024.10.8]** Support for RAG evaluation, including independent evaluation of embedding models and rerankers using [MTEB/CMTEB](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html), as well as end-to-end evaluation using [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html).
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
   pip install evalscope[all]           # Install all backends (Native, OpenCompass, VLMEvalKit)
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
   pip install -e '.[all]'           # Install all backends (Native, OpenCompass, VLMEvalKit)
   ```


## 🚀 Quick Start

### 1. Simple Evaluation
To evaluate a model using default settings on specified datasets, follow the process below:

#### Install using pip
You can execute this command from any directory:
```bash
python -m evalscope.run \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --datasets arc 
```

#### Install from source
Execute this command in the `evalscope` directory:
```bash
python evalscope/run.py \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --datasets arc
```

If prompted with `Do you wish to run the custom code? [y/N]`, please type `y`.


#### Basic Parameter Descriptions
- `--model`: Specifies the `model_id` of the model on [ModelScope](https://modelscope.cn/), allowing automatic download. For example, see the [Qwen2-0.5B-Instruct model link](https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct/summary); you can also use a local path, such as `/path/to/model`.
- `--template-type`: Specifies the template type corresponding to the model. Refer to the `Default Template` field in the [template table](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-datasets.html#llm) for filling in this field.
- `--datasets`: The dataset name, allowing multiple datasets to be specified, separated by spaces; these datasets will be automatically downloaded. Refer to the [supported datasets list](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset.html) for available options.

### 2. Parameterized Evaluation
If you wish to conduct a more customized evaluation, such as modifying model parameters or dataset parameters, you can use the following commands:

**Example 1:**
```shell
python evalscope/run.py \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --model-args revision=master,precision=torch.float16,device_map=auto \
 --datasets gsm8k ceval \
 --use-cache true \
 --limit 10
```

**Example 2:**
```shell
python evalscope/run.py \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --generation-config do_sample=false,temperature=0.0 \
 --datasets ceval \
 --dataset-args '{"ceval": {"few_shot_num": 0, "few_shot_random": false}}' \
 --limit 10
```

#### Parameter Descriptions
In addition to the three [basic parameters](#basic-parameter-descriptions), the other parameters are as follows:
- `--model-args`: Model loading parameters, separated by commas, in `key=value` format.
- `--generation-config`: Generation parameters, separated by commas, in `key=value` format.
  - `do_sample`: Whether to use sampling, default is `false`.
  - `max_new_tokens`: Maximum generation length, default is 1024.
  - `temperature`: Sampling temperature.
  - `top_p`: Sampling threshold.
  - `top_k`: Sampling threshold.
- `--use-cache`: Whether to use local cache, default is `false`. If set to `true`, previously evaluated model and dataset combinations will not be evaluated again, and will be read directly from the local cache.
- `--dataset-args`: Evaluation dataset configuration parameters, provided in JSON format, where the key is the dataset name and the value is the parameter; note that these must correspond one-to-one with the values in `--datasets`.
  - `--few_shot_num`: Number of few-shot examples.
  - `--few_shot_random`: Whether to randomly sample few-shot data; if not specified, defaults to `true`.
- `--limit`: Maximum number of evaluation samples per dataset; if not specified, all will be evaluated, which is useful for quick validation.

### 3. Use the run_task Function to Submit an Evaluation Task
Using the `run_task` function to submit an evaluation task requires the same parameters as the command line. You need to pass a dictionary as the parameter, which includes the following fields:

#### 1. Configuration Task Dictionary Parameters
```python
import torch
from evalscope.constants import DEFAULT_ROOT_CACHE_DIR

# Example
your_task_cfg = {
        'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
        'generation_config': {'do_sample': False, 'repetition_penalty': 1.0, 'max_new_tokens': 512},
        'dataset_args': {},
        'dry_run': False,
        'model': 'qwen/Qwen2-0.5B-Instruct',
        'template_type': 'qwen',
        'datasets': ['arc', 'hellaswag'],
        'work_dir': DEFAULT_ROOT_CACHE_DIR,
        'outputs': DEFAULT_ROOT_CACHE_DIR,
        'mem_cache': False,
        'dataset_hub': 'ModelScope',
        'dataset_dir': DEFAULT_ROOT_CACHE_DIR,
        'limit': 10,
        'debug': False
    }
```
Here, `DEFAULT_ROOT_CACHE_DIR` is set to `'~/.cache/evalscope'`.

#### 2. Execute Task with run_task
```python
from evalscope.run import run_task
run_task(task_cfg=your_task_cfg)
```


## Evaluation Backend
EvalScope supports using third-party evaluation frameworks to initiate evaluation tasks, which we call Evaluation Backend. Currently supported Evaluation Backend includes:
- **Native**: EvalScope's own **default evaluation framework**, supporting various evaluation modes including single model evaluation, arena mode, and baseline model comparison mode.
- [OpenCompass](https://github.com/open-compass/opencompass): Initiate OpenCompass evaluation tasks through EvalScope. Lightweight, easy to customize, supports seamless integration with the LLM fine-tuning framework ms-swift. [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/opencompass_backend.html)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): Initiate VLMEvalKit multimodal evaluation tasks through EvalScope. Supports various multimodal models and datasets, and offers seamless integration with the LLM fine-tuning framework ms-swift. [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: Initiate RAG evaluation tasks through EvalScope, supporting independent evaluation of embedding models and rerankers using [MTEB/CMTEB](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html), as well as end-to-end evaluation using [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html): [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/index.html)
- **ThirdParty**: Third-party evaluation tasks, such as [ToolBench](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html) and [LongBench-Write](https://evalscope.readthedocs.io/en/latest/third_party/longwriter.html).

## Custom Dataset Evaluation
EvalScope supports custom dataset evaluation. For detailed information, please refer to the Custom Dataset Evaluation [📖User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset.html)

## Offline Evaluation
You can use local dataset to evaluate the model without internet connection. 

Refer to: Offline Evaluation [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/offline_evaluation.html)


## Arena Mode
The Arena mode allows multiple candidate models to be evaluated through pairwise battles, and can choose to use the AI Enhanced Auto-Reviewer (AAR) automatic evaluation process or manual evaluation to obtain the evaluation report. 

Refer to: Arena Mode [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html)

## Model Serving Performance Evaluation
A stress testing tool that focuses on large language models and can be customized to support various data set formats and different API protocol formats.

Refer to : Model Serving Performance Evaluation [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test.html)



## TO-DO List
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
- [ ] Auto-reviewer
  - [ ] Qwen-max


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)