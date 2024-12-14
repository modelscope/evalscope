
## Description
The LongWriter supports 10,000+ Word Generation From Long Context LLMs.
We can use the benchmark LongBench-Write focuses more on measuring the long output quality as well as the output length.

GitHub: [LongWriter](https://github.com/THUDM/LongWriter)

Technical Report: [Minimum Tuning to Unlock Long Output from LLMs with High Quality Data as the Key](https://arxiv.org/abs/2410.10210)


## Usage

### Installation

```bash
pip install evalscope[framework] -U
pip install vllm -U
```

### Task configuration

There are few ways to configure the task: dict, json and yaml.

1. Configuration with dict:

```python
task_cfg = dict(stage=['infer', 'eval_l', 'eval_q'],
                model='ZhipuAI/LongWriter-glm4-9b',
                input_data_path=None,
                output_dir='./outputs',
                infer_config={
                    'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions',
                    'is_chat': True,
                    'verbose': False,
                    'generation_kwargs': {
                        'max_new_tokens': 32768,
                        'temperature': 0.5,
                        'repetition_penalty': 1.0
                    },
                    'proc_num': 16,
                },
                eval_config={
                    # No need to set OpenAI info if skipping the stage `eval_q`
                    'openai_api_key': None,
                    'openai_api_base': 'https://api.openai.com/v1/chat/completions',
                    'openai_gpt_model': 'gpt-4o-2024-05-13',
                    'generation_kwargs': {
                        'max_new_tokens': 1024,
                        'temperature': 0.5,
                        'stop': None
                    },
                    'proc_num': 8
                }
            )

```
- Arguments:
  - `stage`: To run multiple stages, `infer`--run the inference process. `eval_l`--run eval length process. `eval_q`--run eval quality process with the model-as-judge.
  - `model`: model id on the ModelScope hub, or local model dir. Refer to [LongWriter-glm4-9b](https://modelscope.cn/models/ZhipuAI/LongWriter-glm4-9b/summary) for more details.
  - `input_data_path`: input data path, default to `None`, it means to use [longbench_write](resources/longbench_write.jsonl)
  - `output_dir`: output root directory.
  - `openai_api_key`: openai_api_key when enabling the stage `eval_q` to use `Model-as-Judge`. Default to None if not needed.
  - `openai_gpt_model`: Judge model name from OpenAI. Default to `gpt-4o-2024-05-13`
  - `generation_kwargs`: The generation configs.
  - `proc_num`: process number for inference and evaluation.


2. Configuration with json (Optional):

```json
{
    "stage": [
        "infer",
        "eval_l",
        "eval_q"
    ],
    "model": "ZhipuAI/LongWriter-glm4-9b",
    "input_data_path": null,
    "output_dir": "./outputs",
    "infer_config": {
        "openai_api_base": "http://127.0.0.1:8000/v1/chat/completions",
        "is_chat": true,
        "verbose": false,
        "generation_kwargs": {
            "max_new_tokens": 32768,
            "temperature": 0.5,
            "repetition_penalty": 1.0
        },
        "proc_num": 16
    },
    "eval_config": {
        "openai_api_key": null,
        "openai_api_base": "https://api.openai.com/v1/chat/completions",
        "openai_gpt_model": "gpt-4o-2024-05-13",
        "generation_kwargs": {
            "max_new_tokens": 1024,
            "temperature": 0.5,
            "stop": null
        },
        "proc_num": 8
    }
}

```
Refer to [default_task.json](default_task.json) for more details.


2. Configuration with yaml (Optional):

```yaml
stage:
  - infer
  - eval_l
  - eval_q
model: "ZhipuAI/LongWriter-glm4-9b"
input_data_path: null
output_dir: "./outputs"
infer_config:
  openai_api_base: "http://127.0.0.1:8000/v1/chat/completions"
  is_chat: true
  verbose: false
  generation_kwargs:
    max_new_tokens: 32768
    temperature: 0.5
    repetition_penalty: 1.0
  proc_num: 16
eval_config:
  openai_api_key: null
  openai_api_base: "https://api.openai.com/v1/chat/completions"
  openai_gpt_model: "gpt-4o-2024-05-13"
  generation_kwargs:
    max_new_tokens: 1024
    temperature: 0.5
    stop: null
  proc_num: 8

```
Refer to [default_task.yaml](default_task.yaml) for more details.


### Run Model Inference
We recommend to use the [vLLM](https://github.com/vllm-project/vllm) to deploy the model.

Environment:
* A100(80G) x 1


To start vLLM server, run the following command:
```shell
CUDA_VISIBLE_DEVICES=0 VLLM_USE_MODELSCOPE=True vllm serve --max-model-len=65536 --gpu_memory_utilization=0.95 --trust-remote-code ZhipuAI/LongWriter-glm4-9b
```
- Arguments:
  - `max-model-len`: The maximum length of the model input.
  - `gpu_memory_utilization`: The GPU memory utilization.
  - `trust-remote-code`: Whether to trust the remote code.
  - `model`: Could be a model id on the ModelScope/HuggingFace hub, or a local model dir.

* Note: You can use multiple GPUs by setting `CUDA_VISIBLE_DEVICES=0,1,2,3` alternatively.


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
