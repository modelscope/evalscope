# GPT-OSS Model Evaluation

On August 6, 2025, OpenAI released two open-source models:

- `gpt-oss-120b` — Suitable for production environments, general-purpose tasks, and scenarios requiring high reasoning capabilities. Can run on a single H100 GPU (117B parameters, including 5.1B activation parameters).
- `gpt-oss-20b` — Suitable for low-latency, local, or specific-use scenarios (21B parameters, including 3.6B activation parameters).

Let’s use the [EvalScope](https://github.com/modelscope/evalscope) model evaluation framework to quickly test the inference speed and benchmark performance of these models.

## Environment Setup

To make model deployment easier and improve inference speed, we use vLLM to launch a web service compatible with the OpenAI API format.

⚠️ **Note**: As of August 6, 2025, vLLM version 0.10.1, which supports the gpt-oss models, has not been officially released yet. You need to install vLLM and gpt-oss dependencies from source. It is recommended to start a new Python 3.12 environment to avoid affecting your existing environment.

1. Create and activate a new conda environment:
```bash
conda create -n gpt_oss_vllm python=3.12
conda activate gpt_oss_vllm
```

2. Install the necessary dependencies:
```bash
# Install PyTorch-nightly and vLLM
pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128
# Install FlashInfer
pip install flashinfer-python==0.2.10
# Install evalscope
pip install evalscope[perf] -U
```

3. Start the model service

We successfully launched the gpt-oss-20b model service on an H20 GPU:

To download the model via ModelScope (recommended):
```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_MODELSCOPE=true vllm serve openai-mirror/gpt-oss-20b --served-model-name gpt-oss-20b --trust_remote_code --port 8801
```

To download the model via HuggingFace:
```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 vllm serve openai/gpt-oss-20b --served-model-name gpt-oss-20b --trust_remote_code --port 8801
```

## Inference Speed Test

We use EvalScope’s inference speed testing feature to evaluate the model’s inference speed.

Test environment:
- GPU: H20-96GB * 1
- vLLM version: 0.10.1+gptoss
- Prompt length: 1024 tokens
- Output length: 1024 tokens

Run the test script:
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

Output:
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


                                    Detailed Performance Metrics                                    
┏━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃      ┃      ┃      Avg ┃      P99 ┃    Gen. ┃      Avg ┃     P99 ┃      Avg ┃     P99 ┃   Success┃
┃Conc. ┃  RPS ┃  Lat.(s) ┃  Lat.(s) ┃  toks/s ┃  TTFT(s) ┃ TTFT(s) ┃  TPOT(s) ┃ TPOT(s) ┃      Rate┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│    1 │ 0.15 │    6.811 │    6.854 │  150.34 │    0.094 │   0.096 │    0.007 │   0.007 │    100.0%│
│   10 │ 0.96 │   10.374 │   10.708 │  986.63 │    0.865 │   1.278 │    0.009 │   0.010 │    100.0%│
│   50 │ 2.47 │   20.222 │   22.612 │ 2529.14 │    2.051 │   5.446 │    0.018 │   0.020 │    100.0%│
│  100 │ 3.37 │   29.570 │   35.594 │ 3455.61 │    2.354 │   6.936 │    0.027 │   0.028 │    100.0%│
└──────┴──────┴──────────┴──────────┴─────────┴──────────┴─────────┴──────────┴─────────┴──────────┘


               Best Performance Configuration               
 Highest RPS         Concurrency 100 (3.37 req/sec)         
 Lowest Latency      Concurrency 1 (6.811 seconds)          

Performance Recommendations:
• The system seems not to have reached its performance bottleneck, try higher concurrency
```

## Benchmark Evaluation

We use EvalScope’s benchmark testing function to evaluate the model’s abilities. Here we use the AIME2025 mathematical reasoning benchmark as an example to test the model’s capabilities.

Run the test script:
```python
from evalscope.constants import EvalType
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='gpt-oss-20b',  # Model name
    api_url='http://127.0.0.1:8801/v1',  # Model service address
    eval_type=EvalType.SERVICE, # Evaluation type, here using service evaluation
    datasets=['aime25'],  # Dataset to test
    generation_config={
        'extra_body': {"reasoning_effort": "high"}  # Model generation parameters, set to high reasoning level
    },
    eval_batch_size=10, # Concurrent batch size
    timeout=60000, # Timeout in seconds
)

run_task(task_cfg=task_cfg)
```

Sample output:

The test result here is 0.8. You can try different model generation parameters and test multiple times to see the results.

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

For more supported benchmarks, please refer to the [EvalScope documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html).

## Result Visualization

EvalScope supports visualizing results so you can see the model’s specific outputs.

```bash
pip install 'evalscope[app]'
evalscope app --lang en
```



## Summary
Through the above steps, we have successfully tested the inference speed and benchmark capabilities of the GPT-OSS model using EvalScope. GPT-OSS performs excellently in both inference speed and benchmarking, making it suitable for production and high-performance scenarios.