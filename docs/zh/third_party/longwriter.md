# LongBench-Write

## 描述
LongWriter 使得LLM能够生成超过10000个单词的内容。我们可以使用基准测试 **LongBench-Write** 来评估长输出的质量以及按需生成相应长度文本的能力。


GitHub: [LongWriter](https://github.com/THUDM/LongWriter)

技术报告： [Minimum Tuning to Unlock Long Output from LLMs with High Quality Data as the Key](https://arxiv.org/abs/2410.10210)


## 使用方式

### 安装

```bash
pip install evalscope[framework] -U
pip install vllm -U
```

### 配置质量评审模型

LongBench-Write 的官方质量评分脚本使用 `gpt-4o-2024-05-13`。OpenAI 已于 2026 年 10 月 23 日从 API 下线该快照。[官方评分脚本](https://github.com/THUDM/LongWriter/blob/main/evaluation/eval_quality.py) · [OpenAI 下线公告](https://developers.openai.com/api/docs/deprecations)

运行 `eval_q` 前，请在 `eval_config` 中配置你有权限调用的评审模型。

- `openai_gpt_model` 填写评审服务提供的模型 ID。
- `openai_api_base` 填写完整的 Chat Completions 请求地址，例如 `https://api.openai.com/v1/chat/completions`。使用其他兼容服务时，请换成该服务的完整请求地址。
- `openai_api_key` 填写该服务的 API 密钥。使用下面的 Python 示例时，运行前请设置环境变量 `JUDGE_API_KEY`；使用 JSON 或 YAML 时，请在本地配置文件里替换 `<judge-api-key>`。

所选服务和模型需接受当前实现发送的 `max_tokens`、`temperature` 和 `stop` 参数，并在 `choices[0].message.content` 中返回包含六项评分的 JSON。建议先用少量样本确认请求和评分解析成功。使用自选 judge 时，请在报告中注明模型 ID，并单独解读质量分。

只评测长度时，从 `stage` 中移除 `eval_q`，并将 `eval_config` 设为空对象。

### 任务配置

支持的任务配置格式: dict, json, yaml.

1. 配置dict:

```python
import os

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
                    'openai_api_key': os.environ['JUDGE_API_KEY'],
                    'openai_api_base': 'https://api.openai.com/v1/chat/completions',
                    'openai_gpt_model': '<judge-model-id>',
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
  - `input_data_path`: input data path, default to `None`, it means to use [longbench_write](https://github.com/modelscope/evalscope/blob/main/evalscope/third_party/longbench_write/resources/longbench_write.jsonl)
  - `output_dir`: output root directory.
  - `generation_kwargs`: The generation configs.
  - `proc_num`: process number for inference and evaluation.


2. 配置json (可选):

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
        "openai_api_key": "<judge-api-key>",
        "openai_api_base": "https://api.openai.com/v1/chat/completions",
        "openai_gpt_model": "<judge-model-id>",
        "generation_kwargs": {
            "max_new_tokens": 1024,
            "temperature": 0.5,
            "stop": null
        },
        "proc_num": 8
    }
}
```
参考 [default_task.json](https://github.com/modelscope/evalscope/blob/main/evalscope/third_party/longbench_write/default_task.json) for more details.


2. 配置yaml (可选):

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
  openai_api_key: "<judge-api-key>"
  openai_api_base: "https://api.openai.com/v1/chat/completions"
  openai_gpt_model: "<judge-model-id>"
  generation_kwargs:
    max_new_tokens: 1024
    temperature: 0.5
    stop: null
  proc_num: 8

```
参考 [default_task.yaml](https://github.com/modelscope/evalscope/blob/main/evalscope/third_party/longbench_write/default_task.yaml) for more details.


### 启动模型预测服务
推荐使用 [vLLM](https://github.com/vllm-project/vllm) 来部署模型服务。

硬件环境:
* A100(80G) x 1


启动vLLM服务：
```shell
CUDA_VISIBLE_DEVICES=0 VLLM_USE_MODELSCOPE=True vllm serve --max-model-len=65536 --gpu_memory_utilization=0.95 --trust-remote-code ZhipuAI/LongWriter-glm4-9b
```
- Arguments:
  - `max-model-len`: The maximum length of the model input.
  - `gpu_memory_utilization`: The GPU memory utilization.
  - `trust-remote-code`: Whether to trust the remote code.
  - `model`: Could be a model id on the ModelScope/HuggingFace hub, or a local model dir.

* 提示：可以使用`CUDA_VISIBLE_DEVICES=0,1,2,3`来配置多GPUs环境。


### 运行评测任务

```python
from evalscope.third_party.longbench_write import run_task

run_task(task_cfg=task_cfg)
```


### 结果和指标
See `eval_length.jsonl` and `eval_quality.jsonl` in the outputs dir.

- Metrics:
  - `score_l`: The average score of the length evaluation.
  - `score_q`: The average score of the quality evaluation.
