# Quick Start

EvalScope is an evaluation framework designed for large models, aiming to provide a simple, easy-to-use, and comprehensive evaluation process. This guide will walk you through various evaluation tasks from simple to complex, helping you get started quickly.

## Start Your First Evaluation

You can evaluate a model on specified datasets using default configurations through either command line or Python code.

### Method 1: Using Command Line

Execute the `evalscope eval` command in any path to start evaluation. This is the most recommended way to get started.

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

**Basic Parameter Description:**
- `--model`: Specify the model's ModelScope ID (e.g., `Qwen/Qwen2.5-0.5B-Instruct`) or local path.
- `--datasets`: Specify one or more dataset names, separated by spaces. For supported datasets, please refer to [Dataset List](./supported_dataset/index.md).
- `--limit`: Maximum number of samples to evaluate per dataset, convenient for quick verification. If not set, all data will be evaluated.

### Method 2: Using Python Code

By calling the `run_task` function with a `TaskConfig` configuration, you can run evaluation in a Python environment. The configuration can be a `TaskConfig` object, Python dictionary, or `yaml`/`json` file path.

::::{tab-set}
:::{tab-item} Using `TaskConfig` Object
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
:::
:::{tab-item} Using Python Dictionary
```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct',
    'datasets': ['gsm8k', 'arc'],
    'limit': 5
}

run_task(task_cfg=task_cfg)
```
:::
:::{tab-item} Using YAML File
```{code-block} yaml
:caption: config.yaml

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
:::
:::{tab-item} Using JSON File
```{code-block} json
:caption: config.json

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
:::
::::

### View Evaluation Results

After evaluation is complete, the terminal will print a score report in the following format:

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

```{seealso}
In addition to command line reports, you can also use our visualization tools to analyze evaluation results in depth.

Reference: [Evaluation Result Visualization](visualization.md)
```

## Advanced Usage

### Customize Evaluation Parameters

 **Example 1. Adjusting Model Loading and Inference Generation Parameters**

You can fine-tune the control of model loading, inference generation, and dataset configuration by passing JSON-formatted strings.

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}' \
 --generation-config '{"do_sample":true,"temperature":0.6,"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0, "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

**Common Parameter Descriptions:**
- `--model-args`: Model loading parameters, such as `revision` (version), `precision` (precision), `device_map` (device mapping).
- `--generation-config`: Inference generation parameters, such as `do_sample` (sampling), `temperature` (temperature), `max_tokens` (maximum length).
- `--dataset-args`: Dataset-specific parameters, keyed by dataset name. For example, `few_shot_num` (number of few-shot examples).

 **Example 2. Adjusting Result Aggregation Methods**

For tasks that require multiple generations (such as math problems), you can specify the number of repeated generations k using the `--repeats` parameter, and adjust the result aggregation method through `dataset-args`, such as `mean_and_vote_at_k`, `mean_and_pass_at_k`, `mean_and_pass^k`.

```shell
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k \
 --limit 10 \
 --dataset-args '{"gsm8k": {"aggregation": "mean_and_vote_at_k"}}' \
 --repeats 5
```

```{seealso}
To view all configurable options, please refer to: [Complete Parameter Description](parameters.md)
```

### Evaluate Online Model APIs

EvalScope supports evaluating model services compatible with OpenAI API format. Simply specify the service address, API Key, and set `eval-type` to `openai_api`.

**1. Start Model Service**

Using vLLM as an example, start a model service:
```shell
# Please install vLLM first: pip install vllm
export VLLM_USE_MODELSCOPE=True
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen2.5 \
  --trust-remote-code \
  --port 8801
```

**2. Run Evaluation**

Use the following command to evaluate the API service:
```shell
evalscope eval \
 --model qwen2.5 \
 --api-url http://127.0.0.1:8801/v1 \
 --api-key EMPTY \
 --eval-type openai_api \
 --datasets gsm8k \
 --limit 10
```

### Using Judge Models for Assessment

For certain subjective or open-ended tasks (such as `simple_qa`), you can specify a powerful "judge model" to evaluate the target model's output.

```python
import os
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

task_cfg = TaskConfig(
    model='qwen2.5-7b-instruct',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type=EvalType.SERVICE,
    datasets=['chinese_simpleqa'],
    limit=5,
    judge_strategy=JudgeStrategy.AUTO,
    judge_model_args={
        'model_id': 'qwen2.5-72b-instruct',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
    }
)

run_task(task_cfg=task_cfg)
```

```{seealso}
For detailed configuration of judge models, please refer to: [Judge Model Parameters](./parameters.md#judge-parameters)
```

### Offline Evaluation

In offline environments, you can use locally cached models and datasets for evaluation.

**1. Prepare Local Datasets**

Datasets are hosted on [ModelScope](https://modelscope.cn/datasets) by default and require internet connection to load. For offline environments, you can use local datasets and models with the following process:

1) First, check the ModelScope ID of the dataset you want to use: Find the **Dataset ID** of the dataset you need in the [Supported Datasets](./supported_dataset/index.md) list, for example, the ID for `mmlu_pro` is `modelscope/MMLU-Pro`.
2) Use the modelscope command to download the dataset: Click "Dataset Files" tab -> Click "Download Dataset" -> Copy the command line
```bash
# Download dataset
modelscope download --dataset modelscope/MMLU-Pro --local_dir ./data/mmlu_pro
```
3) Use the directory `./data/mmlu_pro` as the value for the `local_path` parameter.

**2. Prepare Local Models**

Model files are hosted on the ModelScope Hub and require internet connection to load. When you need to create evaluation tasks in an offline environment, you can download models locally in advance:

For example, use the modelscope command to download the [Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct) model locally:
```bash
modelscope download --model modelscope/Qwen2.5-0.5B-Instruct --local_dir ./model/qwen2.5
```

```{seealso}
For more download options, please refer to [ModelScope Download Guide](https://modelscope.cn/docs/models/download).
```

**3. Run Offline Evaluation**

In the evaluation command, point `--model` to the local model path and specify the dataset's `local_path` through `--dataset-args`.

```shell
evalscope eval \
 --model ./model/qwen2.5 \
 --datasets mmlu_pro \
 --dataset-args '{"mmlu_pro": {"local_path": "./data/mmlu_pro"}}' \
 --limit 10
```

## Migration from v0.1.x

If you previously used v0.1.x version, please note the following major changes after upgrading to v1.0+:

1.  **Local Datasets**: `zip` compressed package format is no longer supported. Please refer to the [Offline Evaluation](#offline-evaluation) guide and use the `local_path` parameter to load local datasets.
2.  **Visualization Compatibility**: The output report format in v1.0 has been updated, and old reports are incompatible with the new visualization tools.
3.  **Inference Parameters**: The model inference parameter `n` has been removed. Please use the `repeats` parameter to control the number of repeated generations.
4.  **Evaluation Process Control**: The `stage` parameter has been removed. Added `rerun_review` parameter to force re-execution of the scoring step when `use_cache=True`.
5.  **Dataset Names**: The `gpqa` dataset has been renamed to `gpqa_diamond` and no longer requires manual specification of `subset_list`.

## Frequently Asked Questions

Encountering issues during evaluation? We've compiled a list of frequently asked questions to help you solve problems.

```{seealso}
Reference: [Frequently Asked Questions](faq.md)
```