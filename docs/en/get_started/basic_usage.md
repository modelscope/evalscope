# Basic Usage

## Simple Evaluation
To evaluate a model using default settings on specified datasets, follow the process below:

`````{tabs}
````{tab} Install using pip
You can execute this command from any directory:
```bash
python -m evalscope.run \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --datasets arc 
```
If prompted with `Do you wish to run the custom code? [y/N]`, please type `y`.
````
````{tab} Install from source
Execute this command in the `evalscope` directory:
```bash
python evalscope/run.py \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --datasets arc
```
If prompted with `Do you wish to run the custom code? [y/N]`, please type `y`.
````
`````

### Basic Parameter Descriptions
- `--model`: Specifies the `model_id` of the model on [ModelScope](https://modelscope.cn/), allowing automatic download. For example, see the [Qwen2-0.5B-Instruct model link](https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct/summary); you can also use a local path, such as `/path/to/model`.
- `--template-type`: Specifies the template type corresponding to the model. Refer to the `Default Template` field in the [template table](https://swift.readthedocs.io/zh-cn/latest/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html#id4) for filling in this field.
    ````{note}
    You can also view the list of `template_type` for models using the following code:
    ``` python
    from evalscope.models.template import TemplateType
    print(TemplateType.get_template_name_list())
    ```
    ````
- `--datasets`: The dataset name, allowing multiple datasets to be specified, separated by spaces; these datasets will be automatically downloaded. Refer to the [supported datasets list](#supported-datasets-list) for available options.

## Parameterized Evaluation
If you wish to conduct a more customized evaluation, such as modifying model parameters or dataset parameters, you can use the following commands:

**Example 1:**
```shell
python evalscope/run.py \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --model-args revision=v1.0.2,precision=torch.float16,device_map=auto \
 --datasets mmlu ceval \
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

### Parameter Descriptions
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

## Use the run_task Function to Submit an Evaluation Task
Using the `run_task` function to submit an evaluation task requires the same parameters as the command line. You need to pass a dictionary as the parameter, which includes the following fields:

### 1. Configuration Task Dictionary Parameters
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

### 2. Execute Task with run_task
```python
from evalscope.run import run_task
run_task(task_cfg=your_task_cfg)
```

## Supported Datasets List
```{note}
The framework currently supports the following datasets. If the dataset you need is not in the list, please submit an issue, or use the [OpenCompass backend](../user_guides/opencompass_backend.md) for evaluation, or use the [VLMEvalKit backend](../user_guides/vlmevalkit_backend.md) for multi-modal model evaluation.
```
| Dataset Name       | Link                                                                                   | Status | Note |
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
| `trivia_qa`        | [trivia_qa](https://modelscope.cn/datasets/modelscope/trivia_qa/summary)               | To be integrated |      |