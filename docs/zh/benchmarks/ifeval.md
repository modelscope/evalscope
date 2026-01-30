# IFEval


## 概述

IFEval（Instruction-Following Eval）是一个用于评估语言模型遵循明确、可验证指令能力的基准测试。它包含一系列带有特定格式、内容或结构要求的提示（prompts），这些要求可以被客观地验证。

## 任务描述

- **任务类型**：指令遵循评估
- **输入**：包含明确、可验证约束条件的提示
- **输出**：完全遵循所有指定指令的响应
- **约束类型**：格式、长度、关键词、结构等

## 主要特点

- 包含约500个提示，涵盖25种可验证的指令类型
- 指令可被客观检查（非主观判断）
- 示例：“写恰好3段文字”、“包含单词X”、“使用项目符号列表”
- 测试模型对指令的理解与遵守能力
- 评估标准无歧义

## 评估说明

- 默认配置采用 **0-shot** 评估方式
- 提供四种评估指标：
  - `prompt_level_strict`：提示中的所有指令都必须被严格遵循
  - `prompt_level_loose`：允许对轻微偏差有一定容忍度
  - `inst_level_strict`：按每条指令计算准确率（严格）
  - `inst_level_loose`：按每条指令计算准确率（宽松）
- 主要指标为 `prompt_level_strict`
- 支持自动验证指令遵循情况

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `ifeval` |
| **数据集ID** | [opencompass/ifeval](https://modelscope.cn/datasets/opencompass/ifeval/summary) |
| **论文** | N/A |
| **标签** | `InstructionFollowing` |
| **指标** | `prompt_level_strict`, `inst_level_strict`, `prompt_level_loose`, `inst_level_loose` |
| **默认示例数（Shots）** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 541 |
| 提示词长度（平均） | 210.75 字符 |
| 提示词长度（最小/最大） | 53 / 1858 字符 |

## 样例示例

**子集**：`default`

```json
{
  "input": [
    {
      "id": "cb71907f",
      "content": "Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "key": 1000,
    "prompt": "Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.",
    "instruction_id_list": [
      "punctuation:no_comma",
      "detectable_format:number_highlighted_sections",
      "length_constraints:number_words"
    ],
    "kwargs": [
      {
        "num_highlights": null,
        "relation": null,
        "num_words": null,
        "num_placeholders": null,
        "prompt_to_repeat": null,
        "num_bullets": null,
        "section_spliter": null,
        "num_sections": null,
        "capital_relation": null,
        "capital_frequency": null,
        "keywords": null,
        "num_paragraphs": null,
        "language": null,
        "let_relation": null,
        "letter": null,
        "let_frequency": null,
        "end_phrase": null,
        "forbidden_words": null,
        "keyword": null,
        "frequency": null,
        "num_sentences": null,
        "postscript_marker": null,
        "first_word": null,
        "nth_paragraph": null
      },
      {
        "num_highlights": 3,
        "relation": null,
        "num_words": null,
        "num_placeholders": null,
        "prompt_to_repeat": null,
        "num_bullets": null,
        "section_spliter": null,
        "num_sections": null,
        "capital_relation": null,
        "capital_frequency": null,
        "keywords": null,
        "num_paragraphs": null,
        "language": null,
        "let_relation": null,
        "letter": null,
        "let_frequency": null,
        "end_phrase": null,
        "forbidden_words": null,
        "keyword": null,
        "frequency": null,
        "num_sentences": null,
        "postscript_marker": null,
        "first_word": null,
        "nth_paragraph": null
      },
      {
        "num_highlights": null,
        "relation": "at least",
        "num_words": 300,
        "num_placeholders": null,
        "prompt_to_repeat": null,
        "num_bullets": null,
        "section_spliter": null,
        "num_sections": null,
        "capital_relation": null,
        "capital_frequency": null,
        "keywords": null,
        "num_paragraphs": null,
        "language": null,
        "let_relation": null,
        "letter": null,
        "let_frequency": null,
        "end_phrase": null,
        "forbidden_words": null,
        "keyword": null,
        "frequency": null,
        "num_sentences": null,
        "postscript_marker": null,
        "first_word": null,
        "nth_paragraph": null
      }
    ]
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ifeval \
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
    datasets=['ifeval'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```