# PhyX-MC


## 概述

PhyX 是首个面向现实、视觉接地场景中物理推理的大规模基准测试。这是其多项选择题（multiple-choice）变体：每个大学水平的物理问题均附带一张图和四个选项，模型需输出正确选项对应的字母。

## 任务描述

- **任务类型**：视觉多项选择物理问题求解
- **输入**：一张图像，以及问题描述、提问和四个带标签的选项
- **输出**：单个选项字母（A、B、C 或 D）
- **领域**：大学水平物理（力学、电磁学、热力学、波动/声学、光学、现代物理）

## 核心特性

- 包含 3,000 道大学水平题目（`test`），覆盖 6 个核心领域和 25 个子领域，每个领域作为独立子集提供；设置 `eval_split='test_mini'` 可选用官方 1,000 题的 testmini 测试集。
- 每道题均基于一张图像，其中包含文本未重复说明的关键信息，因此模型必须结合视觉线索与隐含的物理定律进行推理。
- 涵盖 6 种推理类型（物理模型接地、多公式、空间关系、数值、预测性及隐含条件推理）。
- 采用论文默认的 *Text-DeRedundancy* 输入格式：简化后的问题描述加提问，并附上图像。

- 官方提示词被逐字复现，包括要求仅输出选项字母的指令，以确保评估结果与已发表数据具有可比性。

## 评估说明

- 主要指标：`acc`（准确率），对所有题目取平均值，并分别报告整体及各领域的得分。
- 默认评分方式为官方字符串级匹配：从模型回复中提取选项字母并与标准答案比较，接受如 `D:` 或 `**D**` 等符合提示格式的正确选项标记形式。
- 设置 `judge_strategy='llm'`（配合 `judge_model_args`）可复现官方 LLM 判定模式。仅当无法从回复中提取选项字母时，才调用判别模型，与上游实现一致。
- 图像以内联 base64 形式发送，最大约 5 MB；若所用模型对单张图像有更小的大小限制，请在 `dataset_args` 中设置 `max_image_bytes`。
- 资源链接：[论文](https://arxiv.org/abs/2505.15929) | [GitHub](https://github.com/NastyMarcus/PhyX) | [项目主页](https://killthefullmoon.github.io/projects/PhyX/index.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `phyx_mc` |
| **数据集ID** | [evalscope/PhyX](https://modelscope.cn/datasets/evalscope/PhyX/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2505.15929) |
| **标签** | `MCQ`, `MultiModal`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,000 |
| 提示词长度（平均） | 487.19 字符 |
| 提示词长度（最小/最大） | 178 / 2039 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `mechanics` | 550 | 471.63 | 203 | 1364 |
| `electromagnetism` | 550 | 466.88 | 189 | 1125 |
| `thermodynamics` | 500 | 498.81 | 178 | 1283 |
| `waves_acoustics` | 500 | 492.87 | 196 | 1880 |
| `optics` | 500 | 478.61 | 194 | 1376 |
| `modern_physics` | 400 | 525.59 | 199 | 2039 |

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
      "id": "4334f3a0",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~35.6KB]"
        },
        {
          "text": "A patient with a dislocated shoulder is put into a traction apparatus as shown in figure. The pulls $\\vec{A}$ and $\\vec{B} must combine to produce an outward traction force of 12.8 N on the patient’s arm. How large should these pulls be?Please directly answer the question and provide the correct OPTION LETTER ONLY, e.g., A, B, C, D. OPTION: A: 7.55N B: 5.55N C: 7.65N D: 6.65N"
        }
      ]
    }
  ],
  "target": "A",
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
    --datasets phyx_mc \
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
    datasets=['phyx_mc'],
    dataset_args={
        'phyx_mc': {
            # subset_list: ['mechanics', 'electromagnetism', 'thermodynamics']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
