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
- `--datasets`: Dataset names, supporting multiple datasets separated by spaces. Datasets will be automatically downloaded from ModelScope; refer to the [Dataset List](./supported_dataset.md#1-native-supported-datasets) for supported datasets.
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
 --generation-config '{"do_sample":true,"temperature":0.6,"max_new_tokens":512,"chat_template_kwargs":{"enable_thinking": false}}' \
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
  - `max_new_tokens`: Maximum length of generated tokens
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


## Using Local Datasets and Models

By default, datasets are hosted on [ModelScope](https://modelscope.cn/datasets) and require internet access for loading. If you are in an offline environment, you can use local datasets. The process is as follows:

Assume the current local working path is `/path/to/workdir`.

### Download Dataset to Local

```{important}
Before downloading the dataset, please confirm whether the dataset you want to use is stored in a `zip` file or available on modelscope.
```

#### Download Zip Dataset

Due to historical reasons, some datasets are loaded by executing Python scripts. We have organized these datasets into a `zip` file, which includes the following datasets:
```text
.
├── arc
├── bbh
├── ceval
├── cmmlu
├── competition_math
├── general_qa
├── gsm8k
├── hellaswag
├── humaneval
├── mmlu
├── race
├── trivia_qa
└── truthful_qa
```

For these datasets, execute the following commands:
```shell
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip
unzip data.zip
```
The unzipped datasets will be located in the `/path/to/workdir/data` directory, which will be used as the value for the `local_path` parameter in subsequent steps.

#### Download Modelscope Dataset

For datasets that are not in a `zip` file, such as the [mmlu_pro](https://modelscope.cn/datasets/modelscope/MMLU-Pro) dataset, refer to the dataset address in the [Supported Datasets](./supported_dataset.md#1-native-supported-datasets) document and execute the following commands:

```bash
git lfs install
git clone https://www.modelscope.cn/datasets/modelscope/MMLU-Pro.git
```

Use the directory `/path/to/MMLU-Pro` as the value for the `local_path` parameter.


### Download the Model Locally
Model files are hosted on the ModelScope Hub and require internet access for loading. If you need to create evaluation tasks in an offline environment, you can download the model to your local machine in advance:

For example, use Git to download the Qwen2.5-0.5B-Instruct model locally:
```bash
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B-Instruct.git
```

```{seealso}
[ModelScope Model Download Guide](https://modelscope.cn/docs/models/download)
```

### Execute Evaluation Task
Run the following command to perform the evaluation, passing in the local dataset path and model path. Note that `local_path` must correspond one-to-one with the values in the `--datasets` parameter:

```shell
evalscope eval \
 --model /path/to/workdir/Qwen2.5-0.5B-Instruct \
 --datasets arc \
 --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' \
 --limit 10
```
