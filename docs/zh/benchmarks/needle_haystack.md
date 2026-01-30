# Needle-in-a-Haystack


## 概述

Needle in a Haystack 是一个专注于评估长上下文场景中信息检索能力的基准测试。它用于测试模型在大型文档（haystacks）中查找特定信息（needles）的能力。

## 任务描述

- **任务类型**：长上下文信息检索
- **输入**：包含嵌入目标信息的长文档 + 检索问题
- **输出**：提取的目标信息（needle）
- **领域**：长上下文理解、信息检索

## 主要特性

- 在不同上下文长度（1K–32K+ tokens）下测试检索能力
- 在不同文档深度（0%–100%）下测试检索能力
- 支持英文和中文语料库
- 可通过配置参数生成合成样本
- 可生成性能热力图可视化结果

## 评估说明

- 默认上下文长度：**1,000 至 32,000** tokens（可配置）
- 默认深度百分比：**0% 至 100%**（可配置）
- 主要指标：检索 **准确率（Accuracy）**
- 使用 LLM judge 进行灵活的答案匹配
- 可通过 `extra_params` 配置：needle 内容、上下文长度、深度间隔、分词器等
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/needle_haystack.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `needle_haystack` |
| **数据集ID** | [AI-ModelScope/Needle-in-a-Haystack-Corpus](https://modelscope.cn/datasets/AI-ModelScope/Needle-in-a-Haystack-Corpus/summary) |
| **论文** | 无 |
| **标签** | `LongContext`, `Retrieval` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 200 |
| 提示词长度（平均） | 45063.55 字符 |
| 提示词长度（最小/最大） | 1361 / 137407 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `english` | 100 | 70387.8 | 3898 | 137407 |
| `chinese` | 100 | 19739.3 | 1361 | 37893 |

## 样例示例

**子集**：`english`

```json
{
  "input": [
    {
      "id": "f617334d",
      "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
    },
    {
      "id": "a832dc68",
      "content": "Please read the following text and answer the question below.\n\n<text>\n\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n\nWant to start a startup?  Get funded by\nY Combinator.\n\n\n\n\nJuly 2004(This  ... [TRUNCATED] ... ow do you\nget them to come and work for you?  And then of course there's the\nquestion, how do\n</text>\n\n<question>\nWhat is the best thing to do in San Francisco?\n</question>\n\nDon't give information outside the document or repeat your findings."
    }
  ],
  "target": "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "context": "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n\nWant to start a startup?  Get funded by\nY Combinator.\n\n\n\n\nJuly 2004(This essay is derived from a talk at Oscon 2004.)\nA few months ago I finish ... [TRUNCATED] ... m, we need to understand these\nespecially productive people.  What motivates them?  What do they\nneed to do their jobs?  How do you recognize them? How do you\nget them to come and work for you?  And then of course there's the\nquestion, how do",
    "context_length": 1000,
    "depth_percent": 0
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**系统提示（System Prompt）：**
```text
You are a helpful AI bot that answers questions for a user. Keep your response short and direct
```

**提示模板（Prompt Template）：**
```text
Please read the following text and answer the question below.

<text>
{context}
</text>

<question>
{question}
</question>

Don't give information outside the document or repeat your findings.
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `retrieval_question` | `str` | `What is the best thing to do in San Francisco?` | 用于检索评估的问题。 |
| `needles` | `list[str]` | `['\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n']` | 插入上下文中的事实性 needle 字符串列表。 |
| `context_lengths_min` | `int` | `1000` | 生成合成样本的最小上下文长度（tokens）。 |
| `context_lengths_max` | `int` | `32000` | 生成合成样本的最大上下文长度（tokens）。 |
| `context_lengths_num_intervals` | `int` | `10` | 最小与最大上下文长度之间的间隔数量。 |
| `document_depth_percent_min` | `int` | `0` | needle 插入的最小深度百分比。 |
| `document_depth_percent_max` | `int` | `100` | needle 插入的最大深度百分比。 |
| `document_depth_percent_intervals` | `int` | `10` | 最小与最大深度百分比之间的间隔数量。 |
| `tokenizer_path` | `str` | `Qwen/Qwen3-0.6B` | 用于分词的 tokenizer 模型路径。 |
| `show_score` | `bool` | `False` | 是否在热力图输出图像上显示数值分数。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets needle_haystack \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['needle_haystack'],
    dataset_args={
        'needle_haystack': {
            # subset_list: ['english', 'chinese']  # 可选，评估指定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```