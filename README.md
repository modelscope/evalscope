English | [简体中文](README_zh.md)

## Introduction

Large Language Model (LLMs) evaluation has become a critical process for assessing and improving LLMs. To better support the evaluation of large models, we propose the Eval-Scope framework, which includes the following components and features:

- Pre-configured common benchmark datasets, including: MMLU, CMMLU, C-Eval, GSM8K, ARC, HellaSwag, TruthfulQA, MATH, HumanEval, etc.
- Implementation of common evaluation metrics
- Unified model integration, compatible with the generate and chat interfaces of multiple model series
- Automatic evaluation (evaluator):
  - Automatic evaluation for objective questions
  - Implementation of complex task evaluation using expert models
- Reports of evaluation generating
- Arena mode
- Visualization tools
- Model Inference Performance Evaluation [Tutorial](llmuses/perf/README.md)
- Support for OpenCompass as an Evaluation Backend, featuring advanced encapsulation and task simplification to easily submit tasks to OpenCompass for evaluation.
- Full pipeline support: Seamlessly integrate with SWIFT to easily train and deploy model services, initiate evaluation tasks, view evaluation reports, and achieve an end-to-end large model development process.


Features
- Lightweight, minimizing unnecessary abstractions and configurations
- Easy to customize
  - New datasets can be integrated by simply implementing a single class
  - Models can be hosted on ModelScope, and evaluations can be initiated with just a model id
  - Supports deployment of locally hosted models
- Visualization of evaluation reports
- Rich evaluation metrics
- Model-based automatic evaluation process, supporting multiple evaluation modes
  - Single mode: Expert models score individual models
  - Pairwise-baseline mode: Comparison with baseline models
  - Pairwise (all) mode: Pairwise comparison of all models

## News
- **\[2024.06.29\]** The OpenCompass evaluation backend has been integrated into Eval-Scope, allowing users to easily submit tasks to OpenCompass for evaluation. 🔥🔥🔥
- **\[2024.06.13\]** Eval-Scope has been updated to version 0.3.x, which supports the ModelScope SWIFT framework for LLMs evaluation. 🚀🚀🚀
- **\[2024.06.13\]** We have supported the ToolBench as a third-party evaluation backend for Agents evaluation. 🚀🚀🚀



## Installation
### Install with pip
1. create conda environment
```shell
conda create -n eval-scope python=3.10
conda activate eval-scope
```

2. Install Eval-Scope
```shell
pip install llmuses
```

### Install from source code
1. Download source code
```shell
git clone https://github.com/modelscope/eval-scope.git
```

2. Install dependencies
```shell
cd eval-scope/
pip install -e .
```


## Quick Start

### Simple Evaluation
command line with pip installation:
```shell
python -m llmuses.run --model ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets arc --limit 100
```
command line with source code:
```shell
python llmuses/run.py --model ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets mmlu ceval --limit 10
```
Parameters:
- --model: ModelScope model id, model link: [ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary)

### Evaluation with Model Arguments
```shell
python llmuses/run.py --model ZhipuAI/chatglm3-6b --template-type chatglm3 --model-args revision=v1.0.2,precision=torch.float16,device_map=auto --datasets mmlu ceval --use-cache true --limit 10
```
```
python llmuses/run.py --model qwen/Qwen-1_8B --generation-config do_sample=false,temperature=0.0 --datasets ceval --dataset-args '{"ceval": {"few_shot_num": 0, "few_shot_random": false}}' --limit 10
```
Parameters:
- --model-args: Parameters of model: revision, precision, device_map, in format of key=value,key=value
- --datasets: datasets list, separated by space
- --use-cache: `true` or `false`, whether to use cache, default is `false`
- --dataset-args: evaluation settings，json format，key is the dataset name，value should be args for the dataset
  - --few_shot_num: few-shot data number
  - --few_shot_random: whether to use random few-shot data, default is `true`
  - --local_path: local dataset path
- --limit: maximum number of samples to evaluate for each sub-dataset
- --template-type: model template type, see [Template Type List](https://github.com/modelscope/swift/blob/main/docs/source_en/LLM/Supported-models-datasets.md)

Note: you can use following command to check the template type list of the model:
```shell
from llmuses.models.template import TemplateType
print(TemplateType.get_template_name_list())
```

### Evaluation Backend
Eval-Scope supports using third-party evaluation frameworks to initiate evaluation tasks, which we call Evaluation Backend. Currently supported Evaluation Backend includes:
- **Native**: Eval-Scope：Eval-Scope自身的评测框架，支持多种评估模式，包括单模型评估、竞技场模式、Baseline模型对比模式等。
- [OpenCompass](https://github.com/open-compass/opencompass)：Which is a popular evaluation framework for large language models, Eval-Scope supports submitting tasks to OpenCompass with `pip install ms-opencompass`.
- **ThirdParty**: The third-party task, e.g. [ToolBench](llmuses/thirdparty/toolbench/README.md), you can contribute your own evaluation task to Eval-Scope as third-party backend.

#### 1. OpenCompass Eval-Backend

To facilitate the OpenCompass as an evaluation backend, we have customized the OpenCompass codebase and named it `ms-opencompass`. This version enhances the configuration and execution of evaluation tasks based on the original version and supports installation via PyPI, allowing users to initiate lightweight OpenCompass evaluation tasks through Eval-Scope.


##### Installation
```shell
# Install eval-scope
pip install llmuses>=0.4.0

# Install ms-opencompass
pip install ms-opencompass
```

#### Data Preparation
Available datasets from OpenCompass backend:
```python
'obqa', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada', 'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze', 'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval', 'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench', 'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```
Refer to [OpenCompass datasets](https://hub.opencompass.org.cn/home)

You can use the following code to list all available datasets:
```python
from llmuses.backend.opencompass import OpenCompassBackendManager
print(f'** All datasets from OpenCompass backend: {OpenCompassBackendManager.list_datasets()}')
```

Dataset download:
- Option1: Download from ModelScope
    ```shell
    git clone https://www.modelscope.cn/datasets/swift/evalscope_resource.git
    ```

- Option2: Download from OpenCompass GitHub
    ```shell
    wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
    ```

Unzip the file and set the path to the `data` directory in current work directory.


#### Model serving
We use ModelScope swift to deploy model services, see: [ModelScope swift](https://github.com/modelscope/swift/blob/main/docs/source_en/LLM/VLLM-inference-acceleration-and-deployment.md)
```shell
# Install ms-swift
pip install ms-swift

# Deploy model
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type llama3-8b-instruct --port 8000
```


#### Model evaluation

Refer to example: [example_eval_swift_openai_api](examples/example_eval_swift_openai_api.py) to configure and execute the evaluation task:
```shell
python examples/example_eval_swift_openai_api.py
```



### Local Dataset
You can use local dataset to evaluate the model without internet connection.
#### 1. Download and unzip the dataset
```shell
# set path to /path/to/workdir
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip
unzip data.zip
```


#### 2. Use local dataset to evaluate the model
```shell
python llmuses/run.py --model ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets arc --dataset-hub Local --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}'  --limit 10

# Parameters:
# --dataset-hub: dataset sources: `ModelScope`, `Local`, `HuggingFace` (TO-DO)  default to `ModelScope`
# --dataset-args: json format, key is the dataset name, value should be args for the dataset
```

#### 3. (Optional) Use local mode to submit evaluation task

```shell
# 1. Prepare the model local folder, the folder structure refers to chatglm3-6b, link: https://modelscope.cn/models/ZhipuAI/chatglm3-6b/files
# For example, download the model folder to the local path /path/to/ZhipuAI/chatglm3-6b

# 2. Execute the offline evaluation task
python llmuses/run.py --model /path/to/ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets arc --dataset-hub Local --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' --limit 10
```


### Use run_task function

#### 1. Configuration
```python
import torch
from llmuses.constants import DEFAULT_ROOT_CACHE_DIR

# Example configuration
your_task_cfg = {
        'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
        'generation_config': {'do_sample': False, 'repetition_penalty': 1.0, 'max_new_tokens': 512},
        'dataset_args': {},
        'dry_run': False,
        'model': 'ZhipuAI/chatglm3-6b',
        'template_type': 'chatglm3', 
        'datasets': ['arc', 'hellaswag'],
        'work_dir': DEFAULT_ROOT_CACHE_DIR,
        'outputs': DEFAULT_ROOT_CACHE_DIR,
        'mem_cache': False,
        'dataset_hub': 'ModelScope',
        'dataset_dir': DEFAULT_ROOT_CACHE_DIR,
        'stage': 'all',
        'limit': 10,
        'debug': False
    }

```

#### 2. Execute the task
```python
from llmuses.run import run_task

run_task(task_cfg=your_task_cfg)
```


### Arena Mode
The Arena mode allows multiple candidate models to be evaluated through pairwise battles, and can choose to use the AI Enhanced Auto-Reviewer (AAR) automatic evaluation process or manual evaluation to obtain the evaluation report. The process is as follows:
#### 1. Env preparation
```text
a. Data preparation, the question data format refers to: llmuses/registry/data/question.jsonl
b. If you need to use the automatic evaluation process (AAR), you need to configure the relevant environment variables. Taking the GPT-4 based auto-reviewer process as an example, you need to configure the following environment variables:
    > export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

#### 2. Configuration files
```text
Refer to : llmuses/registry/config/cfg_arena.yaml
Parameters:
    questions_file: question data path
    answers_gen: candidate model prediction result generation, supports multiple models, can control whether to enable the model through the enable parameter
    reviews_gen: evaluation result generation, currently defaults to using GPT-4 as the Auto-reviewer, can control whether to enable this step through the enable parameter
    elo_rating: ELO rating algorithm, can control whether to enable this step through the enable parameter, note that this step depends on the review_file must exist
```

#### 3. Execute the script
```shell
#Usage:
cd llmuses

# dry-run mode
python llmuses/run_arena.py -c registry/config/cfg_arena.yaml --dry-run

# Execute the script
python llmuses/run_arena.py --c registry/config/cfg_arena.yaml
```

#### 4. Visualization

```shell
# Usage:
streamlit run viz.py -- --review-file llmuses/registry/data/qa_browser/battle.jsonl --category-file llmuses/registry/data/qa_browser/category_mapping.yaml
```


### Single Model Evaluation Mode

In this mode, we only score the output of a single model, without pairwise comparison.
#### 1. Configuration file
```text
Refer to: llmuses/registry/config/cfg_single.yaml
Parameters:
    questions_file: question data path
    answers_gen: candidate model prediction result generation, supports multiple models, can control whether to enable the model through the enable parameter
    reviews_gen: evaluation result generation, currently defaults to using GPT-4 as the Auto-reviewer, can control whether to enable this step through the enable parameter
    rating_gen: rating algorithm, can control whether to enable this step through the enable parameter, note that this step depends on the review_file must exist
```
#### 2. Execute the script
```shell
#Example:
python llmuses/run_arena.py --c registry/config/cfg_single.yaml
```

### Baseline Model Comparison Mode

In this mode, we select the baseline model, and compare other models with the baseline model for scoring. This mode can easily add new models to the Leaderboard (just need to run the scoring with the new model and the baseline model).

#### 1. Configuration file
```text
Refer to: llmuses/registry/config/cfg_pairwise_baseline.yaml
Parameters:
    questions_file: question data path
    answers_gen: candidate model prediction result generation, supports multiple models, can control whether to enable the model through the enable parameter
    reviews_gen: evaluation result generation, currently defaults to using GPT-4 as the Auto-reviewer, can control whether to enable this step through the enable parameter
    rating_gen: rating algorithm, can control whether to enable this step through the enable parameter, note that this step depends on the review_file must exist
```
#### 2. Execute the script
```shell
# Example:
python llmuses/run_arena.py --c registry/config/cfg_pairwise_baseline.yaml
```


## Datasets list

| DatasetName        | Link                                                                                   | Status | Note |
|--------------------|----------------------------------------------------------------------------------------|--------|------|
| `mmlu`             | [mmlu](https://modelscope.cn/datasets/modelscope/mmlu/summary)                         | Active |      |
| `ceval`            | [ceval](https://modelscope.cn/datasets/modelscope/ceval-exam/summary)                  | Active |      |
| `gsm8k`            | [gsm8k](https://modelscope.cn/datasets/modelscope/gsm8k/summary)                       | Active |      |
| `arc`              | [arc](https://modelscope.cn/datasets/modelscope/ai2_arc/summary)                       | Active |      |
| `hellaswag`        | [hellaswag](https://modelscope.cn/datasets/modelscope/hellaswag/summary)               | Active |      |
| `truthful_qa`      | [truthful_qa](https://modelscope.cn/datasets/modelscope/truthful_qa/summary)           | Active |      |
| `competition_math` | [competition_math](https://modelscope.cn/datasets/modelscope/competition_math/summary) | Active |      |
| `humaneval`        | [humaneval](https://modelscope.cn/datasets/modelscope/humaneval/summary)               | Active |      |
| `bbh`              | [bbh](https://modelscope.cn/datasets/modelscope/bbh/summary)                           | Active |      |
| `race`             | [race](https://modelscope.cn/datasets/modelscope/race/summary)                         | Active |      |
| `trivia_qa`        | [trivia_qa](https://modelscope.cn/datasets/modelscope/trivia_qa/summary)               | To be intergrated |      |


## Leaderboard
The LLM Leaderboard aims to provide an objective and comprehensive evaluation standard and platform to help researchers and developers understand and compare the performance of models on various tasks on ModelScope.

[Leaderboard](https://modelscope.cn/leaderboard/58/ranking?type=free)



## Experiments and Results
[Experiments](./resources/experiments.md)

## Model Serving Performance Evaluation
[Perf](llmuses/perf/README.md)

## TO-DO List
- ✅Agents evaluation
- [ ] vLLM
- [ ] Distributed evaluating
- [ ] Multi-modal evaluation
- [ ] Benchmarks
  - [ ] GAIA
  - [ ] GPQA
  - [ ] MBPP
- [ ] Auto-reviewer
  - [ ] Qwen-max

