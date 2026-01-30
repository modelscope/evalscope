# DrivelologyBinaryClassification

## 概述

Drivelology 二分类任务评估模型识别“drivelology”的能力——这是一种独特的语言现象，其特征是“有深度的胡言乱语”。这类话语在句法上连贯，但在语用层面具有悖论性、情感负载性或修辞颠覆性。

## 任务描述

- **任务类型**：二分类文本分类（是/否）
- **输入**：待分类的文本样本
- **输出**：若为 drivelology 则输出 "Yes"，否则输出 "No"
- **领域**：语言学分析、幽默检测、语用学

## 主要特点

- 测试对多层次语言含义的理解能力
- 能区分“有深度的胡言乱语”、纯粹的胡言乱语和正常文本
- 需要上下文理解与情感洞察力
- 涵盖幽默、反讽、讽刺等检测
- 提供多个难度级别

## 评估说明

- 默认配置使用 **0-shot** 评估
- 评估指标：准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）
- 子集：binary-english-easy、binary-english-hard、binary-chinese-easy、binary-chinese-hard

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `drivel_binary` |
| **数据集ID** | [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary) |
| **论文** | N/A |
| **标签** | `Yes/No` |
| **指标** | `accuracy`, `precision`, `recall`, `f1_score`, `yes_ratio` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `f1` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,200 |
| 提示词长度（平均） | 1056.08 字符 |
| 提示词长度（最小/最大） | 984 / 1449 字符 |

## 样例示例

**子集**: `binary-classification`

```json
{
  "input": [
    {
      "id": "ddbda8da",
      "content": [
        {
          "text": "#Instruction#:\nClassify whether the given text is a Drivelology sample or not.\n\n#Definition#:\n- Drivelology: Statements that appear logically coherent but contain deeper, often paradoxical meanings.\nThese challenge conventional interpretation ... [TRUNCATED] ... ology.\n\n#Output Format#:\nYou should try your best to answer \"Yes\" if the given input text is Drivelology, otherwise specify \"No\".\nThe answer you give MUST be \"Yes\" or \"No\"\".\n\n#Input Text#: A: Name? B: Henry. A: Age? B: E-N-R-Y.\n#Your Answer#:"
        }
      ]
    }
  ],
  "target": "YES",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "answer": "YES"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

<details>
<summary>少样本（Few-shot）模板</summary>

```text
{question}
```

</details>

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets drivel_binary \
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
    datasets=['drivel_binary'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```