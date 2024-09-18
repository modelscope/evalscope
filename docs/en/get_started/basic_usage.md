# Basic Usage

## 1. Simple Evaluation
To evaluate a model using default settings on specified datasets, follow the process below:

::::{tab-set}

:::{tab-item} Install using pip
You can execute this command from any directory:
```bash
python -m evalscope.run \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --datasets arc 
```
If prompted with `Do you wish to run the custom code? [y/N]`, please type `y`.
:::

:::{tab-item} Install from source
Execute this command in the `evalscope` directory:
```bash
python evalscope/run.py \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --datasets arc
```
If prompted with `Do you wish to run the custom code? [y/N]`, please type `y`.
:::
::::

### Basic Parameter Descriptions
- `--model`: Specifies the `model_id` of the model on [ModelScope](https://modelscope.cn/), allowing automatic download. For example, see the [Qwen2-0.5B-Instruct model link](https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct/summary); you can also use a local path, such as `/path/to/model`.
- `--template-type`: Specifies the template type corresponding to the model. Refer to the `Default Template` field in the [template table](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-datasets.html#llm) for filling in this field.
    ````{note}
    You can also view the list of `template_type` for models using the following code:
    ``` python
    from evalscope.models.template import TemplateType
    print(TemplateType.get_template_name_list())
    ```
    ````
- `--datasets`: The dataset name, allowing multiple datasets to be specified, separated by spaces; these datasets will be automatically downloaded. Refer to the [supported datasets list](./supported_dataset.md#1-native-supported-datasets) for available options.

## 2. Parameterized Evaluation
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

## 3. Use the run_task Function to Submit an Evaluation Task
Using the `run_task` function to submit an evaluation task requires the [same parameters](#parameter-descriptions) as the command line. You need to pass a dictionary as the parameter, which includes the following fields:

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
