# PhyX-OE


## 概述

PhyX 是首个面向现实、视觉接地场景中物理推理的大规模基准测试。这是其开放式变体：不提供选项，模型必须从图示中推导出大学水平物理问题的答案并明确陈述。

## 任务描述

- **任务类型**：视觉开放式物理问题求解
- **输入**：一张图示，加上问题描述和提问
- **输出**：包含逐步推导过程并以最终答案（带单位的数值或公式）结尾的解答
- **领域**：大学水平物理（力学、电磁学、热力学、波动/声学、光学、现代物理）

## 关键特性

- 包含 3,000 道大学水平题目（`test`），覆盖 6 个核心领域和 25 个子领域，每个领域作为独立子集提供；设置 `eval_split='test_mini'` 可选择官方的 1,000 题 testmini 测试集。
- 每道题均基于一张图示，其中包含文本未重复说明的信息，因此模型必须结合视觉线索与隐含的物理定律进行推理。
- 涵盖 6 种推理类型（物理模型接地、多公式、空间关系、数值、预测性及隐含条件推理）。
- 采用论文默认的 *Text-DeRedundancy* 输入风格：简化后的问题描述加提问，并附上图示。

- 官方提示词被逐字复现，包括对逐步推理的要求，以确保评分结果与已发表数据具有可比性。

## 评估说明

- 主要指标：`acc`，即各题目的平均准确率，整体及按领域分别报告。
- 最终答案从 `\boxed{...}` 中提取；若无，则从 'final answer:' / 'correct answer:' 语句中提取；若仍无，则将整个回复与标准答案比较。若回复在给出答案前被截断，则得分为 0（此情况与模型物理能力无关）；请为模型设置足够大的 `generation_config.max_tokens`。
- 答案为自由格式的带单位数值，因此默认使用 LLM 评判器（官方推荐）：设置 `judge.strategy='auto'` 或 `'llm'` 并提供 `judge.models`。仅当答案字符串不完全匹配时才调用评判器。
- 若设置 `judge.strategy='rule'`，则回退到官方的字符串级匹配模式，但会低估准确率，因为等效表达（如 `0.5 m` 与 `50 cm`）在字面上并不相同。
- 图像以内联 base64 形式发送，最大约 5 MB；若所用模型对单张图像有更小的大小限制，请在 `dataset_args` 中设置 `max_image_bytes`。
- 资源链接：[论文](https://arxiv.org/abs/2505.15929) | [GitHub](https://github.com/NastyMarcus/PhyX) | [项目主页](https://killthefullmoon.github.io/projects/PhyX/index.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `phyx_oe` |
| **数据集ID** | [evalscope/PhyX](https://modelscope.cn/datasets/evalscope/PhyX/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2505.15929) |
| **标签** | `MultiModal`, `QA`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,000 |
| 提示词长度（平均） | 364.68 字符 |
| 提示词长度（最小/最大） | 93 / 1874 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `mechanics` | 550 | 356.92 | 124 | 1273 |
| `electromagnetism` | 550 | 326.73 | 107 | 1032 |
| `thermodynamics` | 500 | 390.86 | 93 | 1174 |
| `waves_acoustics` | 500 | 379.95 | 101 | 1731 |
| `optics` | 500 | 361.15 | 109 | 1215 |
| `modern_physics` | 400 | 380.12 | 106 | 1874 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 3,000 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 215x46 - 5712x4953 |
| 格式 | jpeg, png |


## 样例示例

**子集**: `mechanics`

```json
{
  "input": [
    {
      "id": "508a6723",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~35.6KB]"
        },
        {
          "text": "A patient with a dislocated shoulder is put into a traction apparatus as shown in figure. The pulls $\\vec{A}$ and $\\vec{B} must combine to produce an outward traction force of 12.8 N on the patient’s arm. How large should these pulls be? Please answer the question with step by step reasoning."
        }
      ]
    }
  ],
  "target": "7.55N",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": "0",
    "category": "Mechanics",
    "subfield": "Statics",
    "reasoning_type": [
      "Spatial Relation Reasoning"
    ]
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets phyx_oe \
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
    datasets=['phyx_oe'],
    dataset_args={
        'phyx_oe': {
            # subset_list: ['mechanics', 'electromagnetism', 'thermodynamics']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
