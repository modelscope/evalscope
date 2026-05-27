# GPT-OSS 模型评测最佳实践

2025年8月6日，OpenAI 发布了两个开源模型：

- `gpt-oss-120b` —— 适用于生产环境、通用目的和高推理需求的场景，可以在单个 H100 GPU 上运行（117B 参数，其中 5.1B 激活参数）
- `gpt-oss-20b` —— 适用于低延迟和本地或特定用途的场景（21B 参数，其中 3.6B 激活参数）

下面让我们使用[EvalScope](https://github.com/modelscope/evalscope)模型评测框架，来快速测一下模型的推理性能和基准测试能力吧。

## 环境准备

为了更方便的使用模型，并提升推理速度，我们使用 vLLM 启动一个与 OpenAI 格式兼容的 Web 服务。

⚠️ **注意**：由于支持gpt-oss模型的vLLM 0.10.1版本还未正式发布（2025.8.6），需要从源码安装vLLM和gpt-oss的依赖。推荐启动一个新的python 3.12环境，避免对现有环境造成影响。

1. 创建并激活新的conda环境：
```bash
conda create -n gpt_oss_vllm python=3.12
conda activate gpt_oss_vllm
```

2. 安装相关依赖：
```bash
# 安装 PyTorch-nightly 和 vLLM
pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128
# 安装 FlashInfer
pip install flashinfer-python==0.2.10
# 安装 evalscope
pip install evalscope[perf] -U
```

3. 启动模型服务

我们在 H20 GPU上成功启动gpt-oss-20b模型服务：

使用ModelScope下载模型（推荐）：
```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_MODELSCOPE=true vllm serve openai-mirror/gpt-oss-20b --served-model-name gpt-oss-20b --trust_remote_code --port 8801
```

使用HuggingFace下载模型：
```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 vllm serve openai/gpt-oss-20b --served-model-name gpt-oss-20b --trust_remote_code --port 8801
```

## 推理速度测试

我们使用EvalScope的推理速度测试功能，来评测模型的推理速度。

测试环境：
- 显卡: H20-96GB * 1
- vLLM版本: 0.10.1+gptoss
- prompt 长度: 1024 tokens
- 输出长度: 1024 tokens

运行测试脚本：
```bash
evalscope perf \
  --parallel 1 10 50 100 \
  --number 5 20 100 200 \
  --model gpt-oss-20b \
  --url http://127.0.0.1:8801/v1/completions \
  --api openai \
  --dataset random \
  --max-tokens 1024 \
  --min-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 1024 \
  --max-prompt-length 1024 \
  --log-every-n-query 20 \
  --tokenizer-path openai-mirror/gpt-oss-20b \
  --extra-args '{"ignore_eos": true}'
```

输出如下：
```text
╭──────────────────────────────────────────────────────────╮
│ Performance Test Summary Report                          │
╰──────────────────────────────────────────────────────────╯

Basic Information:
┌───────────────────────┬──────────────────────────────────┐
│ Model                 │ gpt-oss-20b                      │
│ Total Generated       │ 332,800.0 tokens                 │
│ Total Test Time       │ 154.57 seconds                   │
│ Avg Output Rate       │ 2153.10 tokens/sec               │
└───────────────────────┴──────────────────────────────────┘


                Performance Overview
┏━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃Conc. ┃ Rate ┃  Num ┃  RPS ┃   Gen/s ┃ Success ┃
┡━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│    1 │    - │  ... │ 0.15 │  150.34 │  100.0% │
│   10 │    - │  ... │ 0.96 │  986.63 │  100.0% │
│   50 │    - │  ... │ 2.47 │ 2529.14 │  100.0% │
│  100 │    - │  ... │ 3.37 │ 3455.61 │  100.0% │
└──────┴──────┴──────┴──────┴─────────┴─────────┘
```

## 基准测试

我们使用 EvalScope 的基准测试功能，来评测模型的能力。这里我们以 AIME2025 这个数学推理基准测试为例，测试模型的能力。

运行测试脚本：
```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='gpt-oss-20b',  # 模型名称
    api_url='http://127.0.0.1:8801/v1',  # 模型服务地址
    eval_type='openai_api', # 评测类型，这里使用openai_api评测
    datasets=['aime25'],  # 测试的数据集
    generation_config={
        'extra_body': {"reasoning_effort": "high"}  # 模型生成参数，这里设置为高推理水平
    },
    eval_batch_size=10, # 并发测试的batch size
    timeout=60000, # 超时时间，单位为秒
)

run_task(task_cfg=task_cfg)
```

输出如下：

这里测试结果为0.8，大家可以尝试不同的模型生成参数，多次测试，查看结果。

```text
+-------------+-----------+---------------+-------------+-------+---------+---------+
| Model       | Dataset   | Metric        | Subset      |   Num |   Score | Cat.0   |
+=============+===========+===============+=============+=======+=========+=========+
| gpt-oss-20b | aime25    | AveragePass@1 | AIME2025-I  |    15 |     0.8 | default |
+-------------+-----------+---------------+-------------+-------+---------+---------+
| gpt-oss-20b | aime25    | AveragePass@1 | AIME2025-II |    15 |     0.8 | default |
+-------------+-----------+---------------+-------------+-------+---------+---------+
| gpt-oss-20b | aime25    | AveragePass@1 | OVERALL     |    30 |     0.8 | -       |
+-------------+-----------+---------------+-------------+-------+---------+---------+ 
```

更多支持的基准测试可以参考[EvalScope文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html)。


## 结果可视化

EvalScope 支持结果可视化，您可以查看模型的具体输出。

```bash
pip install 'evalscope[service]'
evalscope service
```

## 总结
通过以上步骤，我们成功地使用 EvalScope 测试了 GPT-OSS 模型的推理速度和基准测试能力。GPT-OSS 模型在推理速度和基准测试上表现出色，适合用于生产环境和高性能需求的场景。