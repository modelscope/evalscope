# BFCL-v4

## Introduction

The Berkeley Function Calling Leaderboard (BFCL) is specifically designed to evaluate the **function calling capabilities** of Large Language Models (LLMs). Unlike previous evaluations, BFCL considers various forms of function calls, diverse scenarios, and executability. This benchmark has been widely used by multiple research teams (including Llama, Qwen, etc.) and has been tested on numerous large language models. BFCL-v4 is an upgraded version of BFCL-v3, adding agent-related evaluation tasks including web search and model memory.

For more detailed information, please refer to:
- [Web Search Blog](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)
- [Memory Blog](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html)

The benchmark consists of 22 tasks covering various function calling scenarios, with composition and functionality shown in the following diagram:

```python
SUBSET_LIST = [
    # Non-Live Function Calling
    'simple_python',
    'simple_java',
    'simple_javascript',
    'multiple',
    'parallel',
    'parallel_multiple',
    # Live Function Calling
    'live_simple',
    'live_multiple',
    'live_parallel',
    'live_parallel_multiple',
    # Relevance
    'irrelevance',
    'live_irrelevance',
    'live_relevance', # not counted in final score
    # Multi Turn
    'multi_turn_base',
    'multi_turn_miss_func',
    'multi_turn_miss_param',
    'multi_turn_long_context',
    # Web Search
    'web_search_base',
    'web_search_no_snippet',
    # Memory
    'memory_kv',
    'memory_vector',
    'memory_rec_sum'
]
```

![bfcl_v4](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/third_party/bfcl_v4.jpg)

**Overall Score Calculation Formula**:
`Overall Score = (Agentic × 40%) + (Multi-Turn × 30%) + (Live × 10%) + (Non-Live × 10%) + (Hallucination × 10%)`

```{important}
Please note the following when using this evaluation benchmark:
- The `live_relevance` task is not included in the final score.
- When evaluating Web Search tasks, please ensure that `SERPAPI_API_KEY` is set in the environment variables to access the [SerpAPI](https://serpapi.com/dashboard) service, and make sure you have sufficient call credits, otherwise the evaluation results may be all zeros.
- When evaluating Memory tasks, preprocessing requests will run first to populate the memory, which can be time-consuming, so please be patient.
- When evaluating the `memory_vector` task, the required model files will be automatically downloaded from the Hugging Face Hub, so please ensure your network connection is stable.
```

## Installing Dependencies

Before running the evaluation, you need to install the following dependencies:

Install `evalscope`
```bash
# Before the release of v1.1.2, it's recommended to install evalscope from source
git clone https://github.com/modelscope/evalscope.git
cd evalscope/
pip install -e .
```

Install `bfcl-eval`
```bash
pip install bfcl-eval==2025.10.27.1 
```

## Usage

Run the code below to start the evaluation. The following example uses the qwen-plus model for evaluation.

**⚠️ Note: Only API model services are supported. For local model evaluation, it is recommended to use frameworks like vLLM to start the service first.**

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # Using API model service
    datasets=['bfcl_v4'],
    eval_batch_size=10,
    dataset_args={
        'bfcl_v4': {
            # List of evaluation subtasks
            'subset_list': [
                'simple_python',
                'simple_java',
                'simple_javascript',
                'multiple',
                'parallel',
                'parallel_multiple',
                'irrelevance',
                'live_simple',
                'live_multiple',
                'live_parallel',
                'live_parallel_multiple',
                'live_irrelevance',
                'live_relevance',
                'multi_turn_base',
                'multi_turn_miss_func',
                'multi_turn_miss_param',
                'multi_turn_long_context',
                'web_search_base',
                'web_search_no_snippet',
                'memory_kv',
                'memory_vector',
                'memory_rec_sum'
            ],
            'extra_params':{
                # Model refuses to use dots (`.`) in function names; set this to automatically convert dots to underscores during evaluation.
                'underscore_to_dot': True,
                # Whether the model is a Function Calling Model; if true, function calling configuration will be enabled; otherwise, prompts will be used to bypass function calling.
                'is_fc_model': True,
                # When evaluating Web Search tasks, please ensure that SERPAPI_API_KEY is set in the environment variables to access the SerpAPI service.
                'SERPAPI_API_KEY': os.getenv('SERPAPI_API_KEY'),
            }
        }
    },
    generation_config={
        'temperature': 0, # Only temperature parameter is supported, other parameters will be ignored
    },
    use_cache='outputs/bfcl_v4', # Setting a cache directory is recommended to speed up re-runs in case of errors
    limit=10,  # Limit the evaluation count for quick testing; remove this for formal evaluation
)
run_task(task_cfg=task_cfg)
```

Output example:

```text
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| Model     | Dataset   | Metric   | Subset                  |   Num |   Score | Cat.0   |
+===========+===========+==========+=========================+=======+=========+=========+
| qwen-plus | bfcl_v4   | acc      | irrelevance             |    10 |  1      | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | live_irrelevance        |    10 |  0.9    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | live_multiple           |    10 |  0.8    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | live_parallel           |    10 |  0.9    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | live_parallel_multiple  |    10 |  0.7    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | live_relevance          |    10 |  0.8    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | live_simple             |    10 |  1      | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | memory_kv               |    10 |  0.1    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | memory_rec_sum          |    10 |  0.3    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | memory_vector           |    10 |  0.1    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | multi_turn_base         |    10 |  0.5    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | multi_turn_long_context |    10 |  0.7    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | multi_turn_miss_func    |    10 |  0.1    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | multi_turn_miss_param   |    10 |  0.2    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | multiple                |    10 |  0.9    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | parallel                |    10 |  0.9    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | parallel_multiple       |    10 |  0.9    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | simple_java             |    10 |  0.5    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | simple_javascript       |    10 |  0.6    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | simple_python           |    10 |  1      | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | web_search_base         |    10 |  0.7    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | web_search_no_snippet   |    10 |  0.4    | default |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | AGENTIC                 |    50 |  0.3584 | -       |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | MULTI_TURN              |    40 |  0.375  | -       |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | NON_LIVE                |    60 |  0.85   | -       |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | LIVE                    |    40 |  0.85   | -       |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | HALLUCINATION           |    20 |  0.95   | -       |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
| qwen-plus | bfcl_v4   | acc      | OVERALL                 |   210 |  0.5209 | -       |
+-----------+-----------+----------+-------------------------+-------+---------+---------+
```