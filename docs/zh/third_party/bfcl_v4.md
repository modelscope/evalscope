# BFCL-v4

## 简介

Berkeley Function Calling Leaderboard (BFCL) 专门用于评估大型语言模型（LLM）**调用函数的能力**。与之前的评估不同，BFCL 考虑了各种形式的函数调用、多样化的场景以及可执行性。该评测基准被多个研究团队广泛使用（Llama, Qwen等），并且在多个大型语言模型上进行了测试。BFCL-v4 是 BFCL-v3 的升级版本，增加了Agent 相关的评测任务包括网络搜索、模型记忆等。

具体介绍可以参考相关：
- [Web Search 博客](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)
- [Memory 博客](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html)

该评测基准由22个任务组成，涵盖了多种函数调用场景，组成和功能如下图所示：

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

**总分计算公式**：
`Overall Score = (Agentic × 40%) + (Multi-Turn × 30%) + (Live × 10%) + (Non-Live × 10%) + (Hallucination × 10%)`

```{important}
在使用该评测基准时请注意以下几点：
- `live_relevance` 任务不计入最终得分。
- 评测网络搜索（Web Search）任务时，请确保在环境变量中设置 `SERPAPI_API_KEY`，以便访问 [SerpAPI](https://serpapi.com/dashboard) 服务，并确保有充足的调用额度，否则评测结果可能都为0。
- 评测模型记忆（Memory）任务时，会先运行预处理请求以填充记忆，这个过程可能比较耗时，请耐心等待。
- 评测`memory_vector`任务时，会自动从Hugging Face Hub下载所需的模型文件，请确保网络连接正常。
```

## 安装依赖

在运行评测之前需要安装以下依赖：

安装 `evalscope`
```bash
# v1.1.2 版本发布之前，建议从源码安装 evalscope
git clone https://github.com/modelscope/evalscope.git
cd evalscope/
pip install -e .
```

安装 `bfcl-eval`
```bash
pip install bfcl-eval==2025.10.27.1 
```

## 使用方法

运行下面的代码即可启动评测。下面以qwen-plus模型为例进行评测。

**⚠️ 注意：仅支持API模型服务评测，本地模型评测建议使用vLLM等框架预先拉起服务。**

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # 使用API模型服务
    datasets=['bfcl_v4'],
    eval_batch_size=10,
    dataset_args={
        'bfcl_v4': {
            # 评测子任务列表
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
                # 模型在函数名称中拒绝使用点号（`.`）；设置此项，以便在评估期间自动将点号转换为下划线。
                'underscore_to_dot': True,
                # 模式是否为函数调用模型（Function Calling Model），如果是则会启用函数调用相关的配置；否则会使用prompt绕过函数调用。
                'is_fc_model': True,
                # 评测网络搜索（Web Search）任务时，请确保在环境变量中设置 SERPAPI_API_KEY，以便访问 SerpAPI 服务。
                'SERPAPI_API_KEY': os.getenv('SERPAPI_API_KEY'),
            }
        }
    },
    generation_config={
        'temperature': 0, # 只支持设置温度参数，其他参数会被忽略
    },
    use_cache='outputs/bfcl_v4', # 建议设置缓存目录，评测出错时可以加快重跑速度
    limit=10,  # 限制评测数量，便于快速测试，正式评测时建议去掉此项
)
run_task(task_cfg=task_cfg)
```

输出示例：

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