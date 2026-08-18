# MILU


## 概述

MILU（Multi-task Indic Language Understanding Benchmark，多任务印度语言理解基准）是一个全面的评估数据集，用于衡量大语言模型（LLM）在11种印度语言上的表现。该数据集涵盖8个领域和41个主题，结合了翻译而来的通用知识问题以及具有印度文化特色的本地内容。

## 任务描述

- **任务类型**：多语言多项选择题问答（Multilingual Multiple-Choice Question Answering）
- **输入**：用11种语言之一编写的包含四个选项的问题
- **输出**：单个正确答案的字母（A/B/C/D）
- **语言**：英语、孟加拉语、古吉拉特语、印地语、卡纳达语、马拉雅拉姆语、马拉地语、奥里亚语、旁遮普语、泰米尔语、泰卢固语

## 主要特点

- 覆盖8个领域 / 41个主题，包括印度特有的文化、历史和时事内容
- 使用母语编写的问题，而非机器翻译版的MMLU
- 每种语言作为独立的数据集配置，可分别加载

## 评估说明

- 默认配置使用 **0-shot** 评估（测试集 `test`）
- 可通过 `subset_list` 参数指定评估特定语言（例如 `['Hindi', 'Tamil']`），或使用 `limit` 限制样本数量 —— 评估全部11种语言的完整测试集将是一次大规模运行
- 设置 `few_shot_num` > 0 可启用少样本提示（few-shot prompting），示例从 `validation` 验证集中抽取
- 默认从 ModelScope 加载（evalscope 的默认 `dataset_hub`），该数据集在此平台公开且无需访问令牌。若显式将 `dataset_hub` 设为 `huggingface`，请注意 `ai4bharat/MILU` 在 Hugging Face 上是受限制数据集 —— 需先在 huggingface.co 上接受数据集条款，并设置 `HF_TOKEN`（或运行 `huggingface-cli login`）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `milu` |
| **数据集ID** | [ai4bharat/MILU](https://modelscope.cn/datasets/ai4bharat/MILU/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认样本数** | 0-shot |
| **评估集** | `test` |
| **训练集** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 79,608 |
| 提示词长度（平均） | 377.16 字符 |
| 提示词长度（最小/最大） | 223 / 2110 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `English` | 13,535 | 397.01 | 227 | 1930 |
| `Bengali` | 6,637 | 359.93 | 232 | 1828 |
| `Gujarati` | 4,826 | 359.36 | 230 | 1785 |
| `Hindi` | 14,831 | 367.43 | 229 | 1907 |
| `Kannada` | 6,234 | 364.45 | 229 | 1753 |
| `Malayalam` | 4,321 | 388.2 | 239 | 2110 |
| `Marathi` | 6,924 | 394.85 | 223 | 1888 |
| `Odia` | 4,525 | 366.63 | 238 | 1825 |
| `Punjabi` | 4,099 | 364.93 | 234 | 1874 |
| `Tamil` | 6,372 | 382.22 | 230 | 1934 |
| `Telugu` | 7,304 | 384.05 | 233 | 1806 |

## 样例示例

**子集**: `English`

```json
{
  "input": [
    {
      "id": "84726982",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nBakelite is what type of polymer?\n\nA) Thermosetting polymer\nB) Thermoplastic polymer\nC) Fibre\nD) Elastomer"
    }
  ],
  "choices": [
    "Thermosetting polymer",
    "Thermoplastic polymer",
    "Fibre",
    "Elastomer"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "English"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets milu \
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
    datasets=['milu'],
    dataset_args={
        'milu': {
            # subset_list: ['English', 'Bengali', 'Gujarati']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
