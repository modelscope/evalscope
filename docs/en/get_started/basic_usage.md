# Basic Usage

## Simple Evaluation
Evaluate a model on specified datasets using default configurations. This framework supports two ways to initiate evaluation tasks: via command line or using Python code.

### Method 1. Using Command Line

::::{tab-set}
:::{tab-item} Use the `eval` command

Execute the `eval` command from any directory:
```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```
:::

:::{tab-item} Run `run.py`

Execute from the `evalscope` root directory:
```bash
python evalscope/run.py \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```
:::
::::

### Method 2. Using Python Code

When using Python code for evaluation, submit the evaluation task with the `run_task` function by passing in a `TaskConfig` as a parameter. It can also be a Python dictionary, a YAML file path, or a JSON file path, for example:

::::{tab-set}
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

:::{tab-item} Using `TaskConfig`

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

:::{tab-item} Using `yaml` file

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

:::{tab-item} Using `json` file

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


### Basic Parameter Descriptions
- `--model`: Specifies the `model_id` of the model in [ModelScope](https://modelscope.cn/), which can be automatically downloaded, for example, [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/summary); it can also be a local path to the model, e.g., `/path/to/model`.
- `--datasets`: Dataset names, supporting multiple datasets separated by spaces. Datasets will be automatically downloaded from ModelScope; refer to the [Dataset List](./supported_dataset/llm.md) for supported datasets.
- `--limit`: Maximum amount of evaluation data per dataset. If not specified, it defaults to evaluating all data, which can be used for quick validation.


**Output Results**

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


## Complex Evaluation

If you wish to conduct more customized evaluations, such as customizing model parameters or dataset parameters, you can use the following command. The evaluation method is the same as simple evaluation, and below is an example of starting the evaluation using the `eval` command:

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}' \
 --generation-config '{"do_sample":true,"temperature":0.6,"max_tokens":512,"chat_template_kwargs":{"enable_thinking": false}}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0, "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

### Parameter Description
- `--model-args`: Model loading parameters, passed as a JSON string:
  - `revision`: Model version
  - `precision`: Model precision
  - `device_map`: Device allocation for the model
- `--generation-config`: Generation parameters, passed as a JSON string and parsed as a dictionary:
  - `do_sample`: Whether to use sampling
  - `temperature`: Generation temperature
  - `max_tokens`: Maximum length of generated tokens
  - `chat_template_kwargs`: Model inference template parameters
- `--dataset-args`: Settings for the evaluation dataset, passed as a JSON string where the key is the dataset name and the value is the parameters. Note that these need to correspond one-to-one with the values in the `--datasets` parameter:
  - `few_shot_num`: Number of few-shot examples
  - `few_shot_random`: Whether to randomly sample few-shot data; if not set, defaults to `true`


**Output Results**

```text
+------------+-----------+-----------------+----------+-------+---------+---------+
| Model      | Dataset   | Metric          | Subset   |   Num |   Score | Cat.0   |
+============+===========+=================+==========+=======+=========+=========+
| Qwen3-0.6B | gsm8k     | AverageAccuracy | main     |    10 |     0.3 | default |
+------------+-----------+-----------------+----------+-------+---------+---------+ 
```

```{seealso}
Reference: [Full Parameter Description](parameters.md)
```

## Model API Service Evaluation

Specify the model API service address (api_url) and API Key (api_key) to evaluate the deployed model API service. In this case, the `eval-type` parameter must be specified as `service`, for example:

For example, to launch a model service using [vLLM](https://github.com/vllm-project/vllm):

```shell
export VLLM_USE_MODELSCOPE=True && python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --served-model-name qwen2.5 --trust_remote_code --port 8801
```
Then, you can use the following command to evaluate the model API service:
```shell
evalscope eval \
 --model qwen2.5 \
 --api-url http://127.0.0.1:8801/v1 \
 --api-key EMPTY \
 --eval-type service \
 --datasets gsm8k \
 --limit 10
```

## Using the Judge Model

During evaluation, the judge model can be used to assess the output of a model. Some datasets require the use of a judge model for evaluation, such as the `simple_qa` dataset. Use the following command to start the evaluation:

```python
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

task_cfg = TaskConfig(
    model='qwen2.5-7b-instruct',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type=EvalType.SERVICE,
    datasets=[
        # 'simple_qa',
        'chinese_simpleqa',
    ],
    eval_batch_size=5,
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
See also: [Judge Model Parameters](./parameters.md#judge-parameters)
```


## Offline Evaluation

By default, datasets are hosted on [ModelScope](https://modelscope.cn/datasets) and require internet access to load. In offline environments, you can use local datasets and models by following the steps below:

### Using Local Datasets

1. First, check the dataset's address on ModelScope: Refer to [Supported Datasets](./supported_dataset/index.md) to find the **Dataset ID** of the dataset you want to use, for example, [MMLU-Pro](https://modelscope.cn/datasets/modelscope/MMLU-Pro/summary).
2. Download the dataset using the modelscope command: Click on the “Dataset Files” tab -> Click “Download Dataset” -> Copy the command line.
```bash
modelscope download --dataset modelscope/MMLU-Pro --local_dir ./data/mmlu_pro
```
3. Use the directory `./data/mmlu_pro` as the value of the `local_path` parameter when passing it in.

### Using Local Models

Model files are hosted on ModelScope Hub and require internet access to load. If you need to create evaluation tasks in an offline environment, you can download the models in advance:

For example, to download the [Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct) model locally using the modelscope command:
```bash
modelscope download --model modelscope/Qwen2.5-0.5B-Instruct --local_dir ./model/qwen2.5
```

```{seealso}
[ModelScope Model Download Guide](https://modelscope.cn/docs/models/download)
```

### Running Evaluation Tasks

Run the following command to perform the evaluation, passing in the local dataset and model paths. Note that the values for `local_path` must correspond one-to-one with those in the `--datasets` parameter:

```shell
evalscope eval \
 --model ./model/qwen2.5 \
 --datasets mmlu_pro \
 --dataset-args '{"mmlu_pro": {"local_path": "./data/mmlu_pro"}}' \
 --limit 10
```

## Switching to Version v1.0

If you previously used version v0.1x and wish to switch to v1.0, please note the following changes:
1. The evaluation dataset downloaded via zip is no longer supported. For using a local dataset, please refer to [this section](#using-local-datasets).
2. Due to changes in the evaluation output format, the EvalScope visualization feature is not compatible with versions prior to 1.0. You must use reports output by version 1.0 for proper visualization.
3. Due to changes in dataset formats, EvalScope's dataset collection feature does not support datasets created with versions prior to 1.0. You will need to recreate datasets using version 1.0.
4. The `n` parameter in model inference is no longer supported; it has been changed to `repeats`.
5. The `stage` parameter has been removed. A new parameter, `rerun_review`, has been added to control whether evaluations are rerun when `use_cache` is specified.