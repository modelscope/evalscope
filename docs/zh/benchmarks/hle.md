# Humanity's-Last-Exam


## 概述

Humanity's Last Exam (HLE) 是一个全面的语言模型基准测试，包含涵盖广泛学科的 2,500 个问题。该基准由 AI 安全中心（Center for AI Safety）和 Scale AI 联合创建，代表了目前最具挑战性的学术基准之一。

## 任务描述

- **任务类型**：专家级问答
- **输入**：问题及可选图像（14% 为多模态数据）
- **输出**：答案、解释及置信度分数
- **领域分布**：数学 (41%)、物理 (9%)、生物/医学 (11%)、计算机科学/AI (10%)、人文学科 (9%)、工程学 (4%)、化学 (7%)、其他 (9%)

## 主要特点

- 涵盖多个学科的 2,500 个专家级问题
- 14% 的问题需要多模态理解能力
- 24% 为选择题，76% 为短答案精确匹配题
- 问题源自各个学术和专业领域
- 响应格式中包含置信度评分

## 评估说明

- 默认评估使用 **test** 划分
- 主要指标：**Accuracy**（准确率），采用 LLM 作为评判者
- 响应格式包括：Explanation（解释）、Answer（答案）和 Confidence（置信度，0-100%）
- **注意**：对于纯文本模型，请将 `extra_params["include_multi_modal"]` 设置为 `False`
- 使用单个 JSON 对象，包含 C/I 判定结果和推理过程，供 LLM 评判者打分


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hle` |
| **数据集ID** | [cais/hle](https://modelscope.cn/datasets/cais/hle/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `QA` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,500 |
| 提示词长度（平均） | 1029.85 字符 |
| 提示词长度（最小/最大） | 234 / 21341 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `Biology/Medicine` | 280 | 1259.39 | 246 | 13702 |
| `Chemistry` | 165 | 812.72 | 236 | 6942 |
| `Computer Science/AI` | 241 | 1581.02 | 263 | 11529 |
| `Engineering` | 111 | 1620.26 | 250 | 21341 |
| `Humanities/Social Science` | 219 | 1069.39 | 256 | 7028 |
| `Math` | 1,021 | 862.46 | 262 | 8952 |
| `Physics` | 230 | 1027.63 | 257 | 17139 |
| `Other` | 233 | 754.94 | 234 | 13655 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 342 |
| 每个样本的图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 329x12 - 14950x2780 |
| 格式 | gif, jpeg, png, webp |


## 样例示例

**子集**: `Biology/Medicine`

```json
{
  "input": [
    {
      "id": "312e7e8b",
      "content": "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"
    },
    {
      "id": "dfe4cc1b",
      "content": [
        {
          "text": "In a bioinformatics lab, Watterson's estimator (theta) and pi (nucleotide diversity) will be calculated from variant call files which contain human phased samples with only single nucleotide variants present, and there are no completely missi ... [TRUNCATED 1042 chars] ... y pi (nucleotide diversity) is biased.\nC. Both Watterson's estimator (theta) and pi (nucleotide diversity) are biased.\nD. Neither Watterson's estimator (theta) nor pi (nucleotide diversity) are biased.\nE. None of the other answers are correct"
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
    "rationale": "First, we recognize that all single nucleotide variants are included somewhere in the sample. It is given that, across “all samples,” there are no “missing single nucleotide variants.” Further, since “[t]he number of samples is arbitrarily la ... [TRUNCATED 1162 chars] ... fferent genotypes that that position, the analysis would consider these two genomes to have the same nucleotide at the position. This reduces the estimated nucleotide diversity, pi. Therefore, pi would be biased in the circumstance described.",
    "raw_subject": "Bioinformatics",
    "category": "Biology/Medicine",
    "has_image": false
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板:**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `include_multi_modal` | `bool` | `True` | 在评估过程中是否包含多模态（图像）问题。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets hle \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
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
    datasets=['hle'],
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    dataset_args={
        'hle': {
            # subset_list: ['Biology/Medicine', 'Chemistry', 'Computer Science/AI']  # 可选，用于评估特定子集
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
