# MMAU


## 概述

MMAU（Massive Multitask Audio Understanding，大规模多任务音频理解）是一个综合性基准测试，用于评估多模态大语言模型在多种音频任务上的音频理解能力。

## 任务描述

- **任务类型**：音频理解（多项选择题）
- **输入**：包含多项选择题的音频录音
- **输出**：正确答案选项（A/B/C/D）
- **类别**：语音、声音、音乐

## 主要特性

- 大规模音频理解基准测试
- 覆盖多个音频领域（语音、环境声音、音乐）
- 四选项多项选择题格式
- 包含 mini 和完整测试集
- 提供按类别的准确率报告

## 评估说明

- 默认配置使用 **test_mini** 划分
- 主要指标：**准确率**（预测字母的精确匹配）
- 报告整体准确率和各任务类别的准确率
- 提示词中包含思维链（chain-of-thought）指令


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mmau` |
| **数据集ID** | [lmms-lab/mmau](https://modelscope.cn/datasets/lmms-lab/mmau/summary) |
| **论文** | N/A |
| **标签** | `Audio`, `MCQ` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test_mini` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
根据音频内容回答以下多项选择题。你的回复最后一行应采用如下格式：'ANSWER: [LETTER]'（不含引号），其中 [LETTER] 是 {letters} 中的一个。回答前请逐步思考。

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
    --datasets mmau \
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
    datasets=['mmau'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```