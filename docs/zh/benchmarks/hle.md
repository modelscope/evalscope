# Humanity's-Last-Exam

## 概述

Humanity's Last Exam（HLE）是一个综合性语言模型基准测试，包含2,500道涵盖广泛学科的问题。该基准由人工智能安全中心（Center for AI Safety）与Scale AI联合创建，是目前最具挑战性的学术基准之一。

## 任务描述

- **任务类型**：专家级问答
- **输入**：问题（14%为多模态，含图片）
- **输出**：答案、解释及置信度分数
- **领域分布**：数学（41%）、物理（9%）、生物/医学（11%）、计算机科学/AI（10%）、人文（9%）、工程（4%）、化学（7%）、其他（9%）

## 主要特点

- 跨多个学科的2,500道专家级问题
- 14%的问题需要多模态理解能力
- 24%为选择题，76%为简答精确匹配题
- 问题来源涵盖各类学术与专业领域
- 响应格式包含置信度评分

## 评估说明

- 默认使用 **test** 数据划分进行评估
- 主要指标：基于大语言模型（LLM）裁判的 **准确率（Accuracy）**
- 响应格式包括：解释（Explanation）、答案（Answer）和置信度（Confidence，0–100%）
- **注意**：对于纯文本模型，请将 `extra_params["include_multi_modal"]` 设为 `False`
- 使用 GRADE: C/I 格式进行 LLM 裁判评分

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hle` |
| **数据集ID** | [cais/hle](https://modelscope.cn/datasets/cais/hle/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `QA` |
| **指标** | `acc` |
| **默认示例数（Shots）** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,500 |
| 提示词长度（平均） | 1029.85 字符 |
| 提示词长度（最小/最大） | 234 / 21341 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Biology/Medicine` | 280 | 1259.39 | 246 | 13702 |
| `Chemistry` | 165 | 812.72 | 236 | 6942 |
| `Computer Science/AI` | 241 | 1581.02 | 263 | 11529 |
| `Engineering` | 111 | 1620.26 | 250 | 21341 |
| `Humanities/Social Science` | 219 | 1069.39 | 256 | 7028 |
| `Math` | 1,021 | 862.46 | 262 | 8952 |
| `Physics` | 230 | 1027.63 | 257 | 17139 |
| `Other` | 233 | 754.94 | 234 | 13655 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 342 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 329x12 – 14950x2780 |
| 图像格式 | gif, jpeg, png, webp |

## 样例示例

**子集**: `Biology/Medicine`

```json
{
  "input": [
    {
      "id": "906a518f",
      "content": "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"
    },
    {
      "id": "d03d8d4e",
      "content": [
        {
          "text": "In a bioinformatics lab, Watterson's estimator (theta) and pi (nucleotide diversity) will be calculated from variant call files which contain human phased samples with only single nucleotide variants present, and there are no completely missi ... [TRUNCATED] ... y pi (nucleotide diversity) is biased.\nC. Both Watterson's estimator (theta) and pi (nucleotide diversity) are biased.\nD. Neither Watterson's estimator (theta) nor pi (nucleotide diversity) are biased.\nE. None of the other answers are correct"
        }
      ]
    }
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "Biology/Medicine",
  "metadata": {
    "uid": "66e88728ba7d8bc0d5806f3a",
    "author_name": "Scott S",
    "rationale": "First, we recognize that all single nucleotide variants are included somewhere in the sample. It is given that, across “all samples,” there are no “missing single nucleotide variants.” Further, since “[t]he number of samples is arbitrarily la ... [TRUNCATED] ... fferent genotypes that that position, the analysis would consider these two genomes to have the same nucleotide at the position. This reduces the estimated nucleotide diversity, pi. Therefore, pi would be biased in the circumstance described.",
    "raw_subject": "Bioinformatics",
    "category": "Biology/Medicine",
    "has_image": false
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `include_multi_modal` | `bool` | `True` | 评估时是否包含多模态（图像）问题。 |

## 使用方法

### 通过命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets hle \
    --limit 10  # 正式评估时请删除此行
```

### 通过 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['hle'],
    dataset_args={
        'hle': {
            # subset_list: ['Biology/Medicine', 'Chemistry', 'Computer Science/AI']  # 可选，用于评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```