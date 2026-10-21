# MMAU

## 概述

MMAU（大规模多任务音频理解）是一个全面的基准测试，用于评估多模态大型语言模型在多种音频任务中的音频理解能力。

## 任务描述

- **任务类型**：音频理解（多项选择）
- **输入**：带有多个选择题的音频录音
- **输出**：正确答案选项（A/B/C/D）
- **类别**：语音、声音、音乐

## 主要特点

- 大规模音频理解基准测试
- 涵盖多个音频领域（语音、环境声音、音乐）
- 四个选项的多项选择格式
- 包括迷你和完整测试集
- 按类别报告准确性

## 评估说明

- 默认配置使用 **test_mini** 分割
- 主要指标：**Accuracy**（预测字母的完全匹配）
- 报告总体准确性和每项任务类别的准确性
- 提示包括链式思维指令
- 需要支持 `input_audio` 内容部分的 OpenAI 兼容服务器以处理音频输入。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mmau` |
| **数据集ID** | [lmms-lab/mmau](https://modelscope.cn/datasets/lmms-lab/mmau/summary) |
| **论文** | N/A |
| **标签** | `Audio`, `MCQ` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `test_mini` |

## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question based on the audio content. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mmau \
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
    datasets=['mmau'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
