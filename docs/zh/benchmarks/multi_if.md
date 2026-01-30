# Multi-IF


## 概述

Multi-IF 是一个用于评估大语言模型（LLM）在多语言环境下执行多轮指令能力的基准测试。它检验模型在不同语言的多轮对话中遵循复杂指令的能力。

## 任务描述

- **任务类型**：多轮多语言指令遵循
- **输入**：包含指令的多轮对话
- **输出**：遵循给定指令的回复
- **领域**：指令遵循、多语言理解

## 主要特性

- 支持 11 种语言：中文、英语、德语、意大利语、越南语、西班牙语、印地语、葡萄牙语、法语、泰语、俄语
- 多轮对话评估（最多 3 轮）
- 在多语言上下文中测试指令遵循能力
- 提供严格和宽松两种评估指标
- 支持提示词级别和指令级别的评分

## 评估说明

- 默认使用 **train** 数据划分进行评估
- 可配置 `max_turns`（1-3，默认为 3）
- 跟踪四项指标：
  - `prompt_level_strict/loose`：提示词级别的严格/宽松准确率
  - `inst_level_strict/loose`：指令级别的严格/宽松准确率
- 依赖库：nltk、langdetect、emoji（用于中文）、pythainlp（用于泰语）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `multi_if` |
| **数据集 ID** | [facebook/Multi-IF](https://modelscope.cn/datasets/facebook/Multi-IF/summary) |
| **论文** | N/A |
| **标签** | `InstructionFollowing`, `MultiLingual`, `MultiTurn` |
| **指标** | `prompt_level_strict`, `inst_level_strict`, `prompt_level_loose`, `inst_level_loose` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,501 |
| 提示词长度（平均） | 0 字符 |
| 提示词长度（最小/最大） | 0 / 0 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Chinese` | 454 | 0 | 0 | 0 |
| `English` | 909 | 0 | 0 | 0 |
| `Italian` | 493 | 0 | 0 | 0 |
| `Spanish` | 516 | 0 | 0 | 0 |
| `Hindi` | 542 | 0 | 0 | 0 |
| `Portuguese` | 524 | 0 | 0 | 0 |
| `French` | 548 | 0 | 0 | 0 |
| `Russian` | 515 | 0 | 0 | 0 |

## 样例示例

**子集**: `Chinese`

```json
{
  "input": [
    {
      "id": "cb0c68fe",
      "content": ""
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "Chinese",
  "metadata": {
    "turns": null,
    "responses": null,
    "turn_1_prompt": "{\"role\": \"user\", \"content\": \"\\u5199\\u4e00\\u4e2a300+\\u5b57\\u7684\\u603b\\u7ed3\\u5173\\u4e8e\\u7ef4\\u57fa\\u767e\\u79d1\\u9875\\u9762\\\"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\\\"\\uff0c\\u4e0d\\u8981\\u4f7f\\u7528\\u4efb\\u4f55\\u9017\\u53f7\\uff0c\\u5e76\\u4e14\\u81f3\\u5c11\\u7528markdown\\u683c\\u5f0f\\u7a81\\u51fa3\\u4e2a\\u6709\\u6807\\u9898\\u7684\\u90e8\\u5206\\uff0c\\u4f8b\\u5982*\\u7a81\\u51fa\\u90e8\\u52061*\\uff0c*\\u7a81\\u51fa\\u90e8\\u52062*\\uff0c*\\u7a81\\u51fa\\u90e8\\u52063*\\u3002\"}",
    "turn_1_instruction_id_list": "[\"punctuation:no_comma\", \"detectable_format:number_highlighted_sections\", \"length_constraints:number_words\"]",
    "turn_1_kwargs": "[\"{}\", \"{\\\"num_highlights\\\": 3}\", \"{\\\"relation\\\": \\\"at least\\\", \\\"num_words\\\": 300}\"]",
    "turn_2_prompt": "{\"role\": \"user\", \"content\": \"\\u4f60\\u7684\\u56de\\u7b54\\u5e94\\u8be5\\u5305\\u542b\\u4ee5\\u4e0b\\u5173\\u952e\\u8bcd\\uff1a\\u5341\\u5b57\\u519b\\uff0c\\u9ece\\u5df4\\u5ae9\\uff0c\\u7a46\\u65af\\u6797\\u3002\"}",
    "turn_2_instruction_id_list": "[\"punctuation:no_comma\", \"detectable_format:number_highlighted_sections\", \"length_constraints:number_words\", \"keywords:existence\"]",
    "turn_2_kwargs": "[\"{}\", \"{\\\"num_highlights\\\": 3}\", \"{\\\"relation\\\": \\\"at least\\\", \\\"num_words\\\": 300}\", \"{\\\"keywords\\\": [\\\"\\\\u5341\\\\u5b57\\\\u519b\\\", \\\"\\\\u9ece\\\\u5df4\\\\u5ae9\\\", \\\"\\\\u7a46\\\\u65af\\\\u6797\\\"]}\"]",
    "turn_3_prompt": "{\"role\": \"user\", \"content\": \"\\u4f60\\u7684\\u56de\\u7b54\\u5e94\\u8be5\\u4ee5\\u201c\\u8fd9\\u4e2a\\u6982\\u8ff0\\u63d0\\u4f9b\\u4e86\\u5341\\u5b57\\u519b\\u65f6\\u4ee3\\u653f\\u6cbb\\u548c\\u6218\\u4e89\\u7684\\u590d\\u6742\\u6027\\u548c\\u9634\\u8c0b\\u7684\\u6982\\u89c8\\u3002\\u201d\\u8fd9\\u4e2a\\u786e\\u5207\\u7684\\u77ed\\u8bed\\u7ed3\\u5c3e\\uff0c\\u4e0d\\u5141\\u8bb8\\u6709\\u5176\\u4ed6\\u6587\\u5b57\\u51fa\\u73b0\\u5728\\u8fd9\\u4e2a\\u77ed\\u8bed\\u540e\\u9762\\u3002\"}",
    "turn_3_instruction_id_list": "[\"punctuation:no_comma\", \"detectable_format:number_highlighted_sections\", \"length_constraints:number_words\", \"keywords:existence\", \"startend:end_checker\"]",
    "turn_3_kwargs": "[\"{}\", \"{\\\"num_highlights\\\": 3}\", \"{\\\"relation\\\": \\\"at least\\\", \\\"num_words\\\": 300}\", \"{\\\"keywords\\\": [\\\"\\\\u5341\\\\u5b57\\\\u519b\\\", \\\"\\\\u9ece\\\\u5df4\\\\u5ae9\\\", \\\"\\\\u7a46\\\\u65af\\\\u6797\\\"]}\", \"{\\\"end_phrase\\\": \\\"\\\\u8fd9\\\\u4e2a\\\\u6982\\\\u8ff0\\\\u63d0\\\\u4f9b\\\\u4e86\\\\u5341\\\\u5b57\\\\u519b\\\\u65f6\\\\u4ee3\\\\u653f\\\\u6cbb\\\\u548c\\\\u6218\\\\u4e89\\\\u7684\\\\u590d\\\\u6742\\\\u6027\\\\u548c\\\\u9634\\\\u8c0b\\\\u7684\\\\u6982\\\\u89c8\\\\u3002\\\"}\"]",
    "key": "1000:1:zh",
    "turn_index": 0,
    "language": "Chinese"
  }
}
```

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `max_turns` | `int` | `3` | 要评估的最大交互轮数（1-3）。可选值：[1, 2, 3] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets multi_if \
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
    datasets=['multi_if'],
    dataset_args={
        'multi_if': {
            # subset_list: ['Chinese', 'English', 'Italian']  # 可选，评估指定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```