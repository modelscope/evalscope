# Qwen3-Coder+Instruct 模型评测最佳实践

Qwen3的新模型来啦，分别是代码模型[Qwen/Qwen3-Coder-480B-A35B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-Coder-480B-A35B-Instruct)和 通用模型 [Qwen/Qwen3-235B-A22B-Instruct-2507](https://modelscope.cn/models/Qwen/Qwen3-235B-A22B-Instruct-2507)，下面让我们使用[EvalScope](https://github.com/modelscope/evalscope)模型评测框架，来快速测一下这两个模型的性能吧。

## 安装依赖

首先，安装[EvalScope](https://github.com/modelscope/evalscope)模型评估框架：

```shell
pip install 'evalscope[app]' -U
pip install bfcl-eval # 安装 bfcl 评测依赖
```

## 评测Qwen3-Coder模型工具调用能力

我们需要通过OpenAI API兼容的推理服务接入模型能力，以进行评测，这里我们使用DashScope提供的API接口。值得注意的是，EvalScope也支持使用transformers进行模型推理评测，详细信息可参考[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#id2)。

下面是使用BFCL-v3基准测试，来评测Coder模型的工具调用能力，具体配置如下：

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen3-coder-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # 使用API模型服务
    datasets=['bfcl_v3'],
    eval_batch_size=10,
    dataset_args={
        'bfcl_v3': {
            'extra_params':{
                # 模型在函数名称中拒绝使用点号（`.`）；设置此项，以便在评估期间自动将点号转换为下划线。
                'underscore_to_dot': True,
                # 模型是否为函数调用模型（Function Calling Model），如果是则会启用函数调用相关的配置；否则会使用prompt绕过函数调用。
                'is_fc_model': True,
            }
        }
    },
    generation_config={
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'repetition_penalty': 1.05,
        'max_tokens': 65536,  # 设置最大生成长度
        'parallel_tool_calls': True,  # 启用并行函数调用
    },
    # limit=50,  # 限制评测数量，便于快速测试，正式评测时建议去掉此项
    ignore_errors=True,  # 忽略错误，可能会被模型拒绝的测试用例
)
run_task(task_cfg=task_cfg)
```

输出结果如下：

可以看到模型整体的工具调用能力较强，但在多轮工具调用以及并行工具调用两方面仍有较大提升空间。

```text
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| Model            | Dataset   | Metric          | Subset                  |   Num |   Score | Cat.0        |
+==================+===========+=================+=========================+=======+=========+==============+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | live_simple             |   257 |  0.8171 | AST_LIVE     |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | live_multiple           |  1039 |  0.8085 | AST_LIVE     |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | live_parallel           |    16 |  0.375  | AST_LIVE     |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | live_parallel_multiple  |    24 |  0.4167 | AST_LIVE     |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | simple                  |   400 |  0.955  | AST_NON_LIVE |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | multiple                |   200 |  0.945  | AST_NON_LIVE |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | parallel                |   200 |  0.55   | AST_NON_LIVE |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | parallel_multiple       |   200 |  0.56   | AST_NON_LIVE |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | java                    |   100 |  0.64   | AST_NON_LIVE |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | javascript              |    50 |  0.82   | AST_NON_LIVE |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | multi_turn_base         |   200 |  0.43   | MULTI_TURN   |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | multi_turn_miss_func    |   200 |  0.24   | MULTI_TURN   |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | multi_turn_miss_param   |   200 |  0.305  | MULTI_TURN   |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | multi_turn_long_context |   200 |  0.385  | MULTI_TURN   |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | irrelevance             |   240 |  0.8458 | RELEVANCE    |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | live_relevance          |    17 |  0.6471 | RELEVANCE    |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | live_irrelevance        |   881 |  0.8343 | RELEVANCE    |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+
| qwen3-coder-plus | bfcl_v3   | AverageAccuracy | OVERALL                 |  4424 |  0.7199 | -            |
+------------------+-----------+-----------------+-------------------------+-------+---------+--------------+ 
```

## 评测Qwen3-Instruct模型知识和推理能力

下面是使用simple\_qa和chinese\_simpleqa两个基准测试，评测模型的知识水平，同时使用Qwen2.5-72B模型评价答案是否正确；使用AIME25来测试模型的复杂推理能力。具体配置如下：

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen3-235b-a22b-instruct-2507',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # 使用API模型服务
    datasets=['simple_qa', 'chinese_simpleqa', 'aime25'],
    eval_batch_size=10,
    generation_config={
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'max_tokens': 16384,  # 设置最大生成长度
    },
    # limit=20,  # 限制评测数量，便于快速测试，正式评测时建议去掉此项
    ignore_errors=True,  # 忽略错误，可能会被模型拒绝的测试用例
    stream=True,  # 启用流式输出
    judge_model_args={ # 配置Judge模型参数
        'model_id': 'qwen2.5-72b-instruct',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 4096
        }
    },
)

run_task(task_cfg=task_cfg)    
   
```

输出结果如下：

可以看出模型展现了良好的推理能力以及较高的知识水平。

```text
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| Model                         | Dataset          | Metric           | Subset               |   Num |   Score | Cat.0   |
+===============================+==================+==================+======================+=======+=========+=========+
| qwen3-235b-a22b-instruct-2507 | aime25           | AveragePass@1    | AIME2025-I           |    15 |  0.6667 | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | aime25           | AveragePass@1    | AIME2025-II          |    15 |  0.6667 | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | aime25           | AveragePass@1    | OVERALL              |    30 |  0.6667 | -       |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_correct       | 中华文化             |    20 |  0.65   | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_correct       | 人文与社会科学       |    20 |  1      | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_correct       | 工程、技术与应用科学 |    20 |  0.8    | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_correct       | 生活、艺术与文化     |    20 |  0.8    | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_correct       | 社会                 |    20 |  0.9    | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_correct       | 自然与自然科学       |    20 |  0.8    | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_correct       | OVERALL              |   120 |  0.825  | -       |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_incorrect     | 中华文化             |    20 |  0.35   | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_incorrect     | 人文与社会科学       |    20 |  0      | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_incorrect     | 工程、技术与应用科学 |    20 |  0.2    | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_incorrect     | 生活、艺术与文化     |    20 |  0.2    | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_incorrect     | 社会                 |    20 |  0.1    | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_incorrect     | 自然与自然科学       |    20 |  0.2    | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_incorrect     | OVERALL              |   120 |  0.175  | -       |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_not_attempted | 中华文化             |    20 |  0      | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_not_attempted | 人文与社会科学       |    20 |  0      | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_not_attempted | 工程、技术与应用科学 |    20 |  0      | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_not_attempted | 生活、艺术与文化     |    20 |  0      | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_not_attempted | 社会                 |    20 |  0      | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_not_attempted | 自然与自然科学       |    20 |  0      | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | chinese_simpleqa | is_not_attempted | OVERALL              |   120 |  0      | -       |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | simple_qa        | is_correct       | default              |    20 |  0.6    | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | simple_qa        | is_incorrect     | default              |    20 |  0.35   | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+
| qwen3-235b-a22b-instruct-2507 | simple_qa        | is_not_attempted | default              |    20 |  0.05   | default |
+-------------------------------+------------------+------------------+----------------------+-------+---------+---------+ 

```

**更多支持的Benchmark请查看**[**文档**](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html)**。**

## 结果可视化

EvalScope支持可视化结果，可以查看模型具体的输出。

运行以下命令，可以启动基于Gradio的可视化界面：

```shell
evalscope app
```

选择评测报告，点击加载，即可看到模型在每个问题上的输出结果，以及整体答题正确率：

![image.png](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/qwen_coder_overview.png)

![image.png](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/qwen_coder_detail.png)

## 总结

本文介绍了使用EvalScope框架评测Qwen3-Coder和Qwen3-Instruct两个新模型性能的方法与结果。评测内容包括：

*   **Qwen3-Coder模型**：使用BFCL-v3基准测试其工具调用能力，结果显示其整体表现较强，但在多轮和并行调用方面仍有提升空间。
    
*   **Qwen3-Instruct模型**：通过simple\_qa、chinese\_simpleqa和AIME25测试其知识与推理能力，结果表现出色。
    

完整评测流程与文档详见[EvalScope官方文档](https://evalscope.readthedocs.io/)。