
## Description
The LongWriter supports 10,000+ Word Generation From Long Context LLMs.
We can use the benchmark LongBench-Write focuses more on measuring the long output quality as well as the output length.

Refer to https://github.com/THUDM/LongWriter

## Usage

### Installation

```bash
pip install evalscope[framework]
```

### Task configuration

There are few ways to configure the task: dict, json and yaml.

1. Configuration with dict:

```python
task_cfg = dict(stage=['infer', 'eval_l', 'eval_q'],
                model='ZhipuAI/LongWriter-glm4-9b',
                input_data_path=None,
                output_dir='./outputs',
                openai_api_key=None,
                openai_gpt_model='gpt-4o-2024-05-13',
                infer_generation_kwargs={
                    'max_new_tokens': 32768,
                    'temperature': 0.5
                },
                eval_generation_kwargs={
                    'max_new_tokens': 1024,
                    'temperature': 0.5,
                    'stop': None
                },
                proc_num=8)

```
- Arguments:
  - `stage`: To run multiple stages, `infer`--run the inference process. `eval_l`--run eval length process. `eval_q`--run eval quality process.
  - `model`: model id on the ModelScope hub, or local model dir.
  - `input_data_path`: input data path, default to `None`, it means to use [longbench_write](resources/longbench_write.jsonl)
  - `output_dir`: output root directory.
  - `openai_api_key`: openai_api_key when enabling the stage `eval_q` to use `Model-as-Judge`. Default to None if not needed.
  - `openai_gpt_model`: Judge model name from OpenAI. Default to `gpt-4o-2024-05-13`
  - `infer_generation_kwargs`: The generation kwargs for models to be evaluated.
  - `eval_generation_kwargs`: The generation kwargs for judge-models.
  - `proc_num`: proc num.


2. Configuration with json (Optional):

```json
{
    "stage": ["infer", "eval_l", "eval_q"],
    "model": "ZhipuAI/LongWriter-glm4-9b",
    "input_data_path": null,
    "output_dir": "./outputs",
    "openai_api_key": null,
    "openai_gpt_model": "gpt-4o-2024-05-13",
    "infer_generation_kwargs": {
        "max_new_tokens": 32768,
        "temperature": 0.5
    },
    "eval_generation_kwargs": {
        "max_new_tokens": 1024,
        "temperature": 0.5,
        "stop": null
    },
    "proc_num": 8
}
```
Refer to [default_task.json](default_task.json) for more details.


2. Configuration with yaml (Optional):

```yaml
stage:
  - infer
  - eval_l
  - eval_q
model: ZhipuAI/LongWriter-glm4-9b
input_data_path: null
output_dir: ./outputs
openai_api_key: null
openai_gpt_model: gpt-4o-2024-05-13
infer_generation_kwargs:
  max_new_tokens: 32768
  temperature: 0.5
eval_generation_kwargs:
  max_new_tokens: 1024
  temperature: 0.5
  stop: null
proc_num: 8

```
Refer to [default_task.yaml](default_task.yaml) for more details.



### Run the task

```python
from evalscope.third_party.longbench_write import run_task

run_task(task_cfg=task_cfg)
```


### Results and metrics
See `eval_length.jsonl` and `eval_quality.jsonl` in the outputs dir.

- Metrics:
  - `score_l`: The average score of the length evaluation.
  - `score_q`: The average score of the quality evaluation.
