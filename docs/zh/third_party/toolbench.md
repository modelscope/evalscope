# ToolBench

## 描述
我们评估ToolBench基准的有效性：[ToolBench](https://arxiv.org/pdf/2307.16789) (Qin et al.,2023b)。任务涉及集成API调用以完成任务，Agent必须准确选择合适的API并构建必要的API请求。

此外，我们根据测试实例中使用的工具在训练期间是否被看到，将ToolBench的测试集划分为域内（in domain）和域外（out of domain）。此划分使我们能够评估在分布内和分布外场景中的性能。我们将此数据集称为`ToolBench-Static`。

有关更多详细信息，请参阅：[Small LLMs Are Weak Tool Learners: A Multi-LLM Agent](https://arxiv.org/abs/2401.07324)

## 数据集

数据集[链接](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static/dataPeview)

- 数据集统计信息：
  - 域内数量：1588
  - 域外数量：781

## 使用方法

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-1.7B',
    datasets=['tool_bench'],
    limit=5,
    eval_batch_size=5,
    generation_config={
        'max_tokens': 1000,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.7,  # 采样温度 (qwen 报告推荐值)
        'top_p': 0.8,  # top-p采样 (qwen 报告推荐值)
        'top_k': 20,  # top-k采样 (qwen 报告推荐值)
        'extra_body':{'chat_template_kwargs': {'enable_thinking': False}}  # 关闭思考模式
    }
)

run_task(task_cfg=task_cfg)
```

结果如下(部分脏数据被自动丢弃，导致in_domain样本数量减少)：
```text
+------------+------------+-----------+---------------+-------+---------+---------+
| Model      | Dataset    | Metric    | Subset        |   Num |   Score | Cat.0   |
+============+============+===========+===============+=======+=========+=========+
| Qwen3-1.7B | tool_bench | Act.EM    | in_domain     |     2 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | Act.EM    | out_of_domain |     5 |  0.2    | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | Plan.EM   | in_domain     |     0 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | Plan.EM   | out_of_domain |     0 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | F1        | in_domain     |     2 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | F1        | out_of_domain |     5 |  0.2    | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | HalluRate | in_domain     |     2 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | HalluRate | out_of_domain |     5 |  0.4    | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | Rouge-L   | in_domain     |     2 |  0      | default |
+------------+------------+-----------+---------------+-------+---------+---------+
| Qwen3-1.7B | tool_bench | Rouge-L   | out_of_domain |     5 |  0.1718 | default |
+------------+------------+-----------+---------------+-------+---------+---------+ 
```

## 结果和指标

- 指标：
  - `Plan.EM`：代理在每一步使用工具调用、生成答案或放弃的计划决策。精确匹配得分。
  - `Act.EM`：动作精确匹配得分，包括工具名称和参数。
  - `HalluRate`（越低越好）：代理在每一步回答时的幻觉率。
  - `Avg.F1`：代理在每一步调用工具的平均F1得分。
  - `Rouge-L`：代理在每一步回答的Rouge-L得分。

通常，我们关注`Act.EM`、`HalluRate`和`Avg.F1`指标。