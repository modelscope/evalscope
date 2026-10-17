# STS-B Decision

## 概述

句子相似度在0到5六个有序级别上进行评分；保留连续的黄金标准。

## 任务描述

- **任务类型**：带类型的文本决策
- **输入**：陈述和一个带有命名选项或有序级别的问题
- **输出**：选择、是的概率或序数得分分布
- **领域**：文本理解和决策

## 主要特点

- 使用ModelScope或Hugging Face上的`sentence-transformers/stsb`公共测试分割
- 将源相似度标签从[0, 1]转换为[0, 5]，保留小数黄金值
- 保持选项顺序、源ID和评分描述
- 支持System One和在其声明能力范围内的原生AutoTrust API

## 评估说明

- 零样本确定性评分；黄金答案和来源从未进入请求
- 主要指标：预期得分值的MAE；同时计算跨样本的Spearman和Pearson相关性
- 不对连续黄金标准进行离散化以计算准确率或排名概率得分
- 无效响应被排除在普通指标之外，并反映在执行覆盖率中
- 六个级别（0-5）适用于两种协议；常量或不足的输入使相关性未定义

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `stsb` |
| **数据集ID** | [sentence-transformers/stsb](https://modelscope.cn/datasets/sentence-transformers/stsb/summary) |
| **论文** | N/A |
| **标签** | `Decision` |
| **指标** | `mae` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,379 |
| 提示词长度（平均） | 391.35 字符 |
| 提示词长度（最小/最大） | 314 / 674 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "8eaa765d",
      "content": "{\"state\": {\"sentence1\": \"A girl is styling her hair.\", \"sentence2\": \"A girl is brushing her hair.\"}, \"question\": {\"type\": \"score\", \"instructions\": \"Rate the semantic similarity of sentence1 and sentence2.\", \"criteria\": [\"Unrelated\", \"Very different\", \"Some shared meaning\", \"Mostly similar\", \"Nearly equivalent\", \"Completely equivalent\"]}}",
      "internal": {
        "systemone": {
          "kind": "score",
          "state": {
            "sentence1": "A girl is styling her hair.",
            "sentence2": "A girl is brushing her hair."
          },
          "instructions": "Rate the semantic similarity of sentence1 and sentence2.",
          "criteria": [
            "Unrelated",
            "Very different",
            "Some shared meaning",
            "Mostly similar",
            "Nearly equivalent",
            "Completely equivalent"
          ],
          "answer_prefix": "ANSWER: "
        }
      }
    }
  ],
  "target": "2.5",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "source_id": null,
    "decision_gold": 2.5,
    "decision_kind": "score",
    "source_group": null,
    "family": "",
    "tier": "",
    "language": "",
    "continuous": true,
    "decision_labels": [
      "0",
      "1",
      "2",
      "3",
      "4",
      "5"
    ],
    "fixed_classes": false
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --eval-type systemone_api \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets stsb \
    --limit 10  # Remove this line for formal evaluation
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='systemone_api',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['stsb'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
