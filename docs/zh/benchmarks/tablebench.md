# TableBench


## 概述

TableBench 评估在现实场景中的表格问答：在四个类型（事实核查、数值推理、数据分析和可视化）的18个子类别中，对真实表格提出了886个问题。此适配器涵盖了前三种类型的836个文本答案问题。

## 任务描述

- **任务类型**：表格问答
- **输入**：包含JSON格式表格的官方指令和问题
- **输出**：推理（根据指令类型），然后是`Final Answer: ...`
- **领域**：来自金融、体育、人口统计学、科学等领域的表格

## 主要特点

- 事实核查（96个）、数值推理（397个：聚合、算术、比较、计数、排名、多跳、基于时间、特定领域）和数据分析（343个：相关性、趋势预测、统计、影响、异常、因果、描述性分析）
- 使用2025年4月18日发布的清理后的测试集和改进后的官方指标
- 三种官方提示风格：`TCoT`（文本链式思维，默认）、`SCoT`（符号链式思维）和`DP`（直接提示）

## 评估说明

- `mix_metric`（主要）是官方的Mix_Metric：每个问题都按照其子类型由官方脚本分配的指标进行评分，然后平均得分：
  - 事实核查、数值推理和影响分析使用EM
  - 相关性分析、趋势预测和统计分析使用带有10%数字容差的EM
  - 其他数据分析子类型使用ROUGE-L
- 答案是该行第一个`Final Answer: `之后的文本，与官方解析器一致
- 与官方代码一样，在评分之前，预测和参考答案会被标准化（小写、去除标点符号和冠词），因此不比较小数点、符号和百分号
- 通过`extra_params={'instruction_type': 'DP'}`选择提示风格
- 可视化（50个图表生成问题）需要执行模型编写的绘图代码，目前尚未包括
- [论文](https://arxiv.org/abs/2408.09174) | [代码](https://github.com/TableBench/TableBench) |
  [数据集](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench) |
  [排行榜](https://tablebench.github.io/)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tablebench` |
| **数据集ID** | [Multilingual-Multimodal-NLP/TableBench](https://modelscope.cn/datasets/Multilingual-Multimodal-NLP/TableBench/summary) |
| **论文** | [论文](https://arxiv.org/abs/2408.09174) |
| **标签** | `QA`, `Reasoning` |
| **指标** | `mix_metric` |
| **默认示例** | 0-shot |
| **评估分割** | `test` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `instruction_type` | `str` | `TCoT` | 官方提示风格：TCoT（文本链式思维）、SCoT（符号链式思维）或DP（直接回答）。选项：['TCoT', 'SCoT', 'DP'] |

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tablebench \
    --limit 10  # Remove this line for formal evaluation
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['tablebench'],
    dataset_args={
        'tablebench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
