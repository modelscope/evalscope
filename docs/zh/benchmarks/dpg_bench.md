# DPG-Bench


## 概述

DPG-Bench 评估生成的图像在多大程度上遵循了关于对象、属性和关系的密集提示词。

## 任务描述

- **任务类型**：文本到图像生成
- **输入**：描述多对象场景的密集提示词
- **输出**：通过从提示词分解出的原子级是/否问题进行评分的图像
- **领域**：密集提示词语义对齐

## 主要特点

- 包含 1,065 个官方提示词和 14,392 个具有依赖关系的是/否问题。
- ModelScope 镜像包含了确切的提示词。若需本地 CSV，请将原始文本文件放置在 `prompts/` 目录下。

## 评估说明

- 使用 `--judge` 配置视觉评判器。缺失或无效的判定结果将排除该图像。官方评判器为 mPLUG。
- 总体分数是对经过依赖调整后的图像分数取平均值。类别分数则汇总所有图像的原始判定结果。
- EvalScope 保留所有 CSV 行，并使用每张图像进行类别评分。参考脚本会跳过其第一行 CSV，并使用最终裁剪区域进行类别评分。使用 `repeats=4` 可为每个提示词生成四张图像。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `dpg_bench` |
| **数据集ID** | [evalscope/DPG-Bench](https://modelscope.cn/datasets/evalscope/DPG-Bench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2403.05135) |
| **标签** | `TextToImage` |
| **指标** | `dpg_score` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,065 |
| 提示词长度（平均） | 407.9 字符 |
| 提示词长度（最小/最大） | 234 / 891 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "27494d92",
      "content": "An empty space where an invisible man would be, with a pair of horn-rimmed glasses seemingly floating in mid-air, and a pearl bead necklace draped in the space below them. In the space where his hands would be, a smartphone is held, as if being operated by the unseen figure. Around this curious scene, the room appears ordinary, with a couch and a coffee table nearby, upon which rests a scattering of magazines and a remote control."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "partiprompts97",
    "prompt": "An empty space where an invisible man would be, with a pair of horn-rimmed glasses seemingly floating in mid-air, and a pearl bead necklace draped in the space below them. In the space where his hands would be, a smartphone is held, as if being operated by the unseen figure. Around this curious scene, the room appears ordinary, with a couch and a coffee table nearby, upon which rests a scattering of magazines and a remote control.",
    "questions": [
      {
        "proposition_id": 1,
        "question": "Is there an empty space?",
        "dependency": [
          0
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 2,
        "question": "Is there an invisible man?",
        "dependency": [
          1
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 3,
        "question": "Are there glasses?",
        "dependency": [
          1
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 4,
        "question": "Is there a necklace?",
        "dependency": [
          1
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 5,
        "question": "Is there a smartphone?",
        "dependency": [
          1
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 6,
        "question": "Is there a room?",
        "dependency": [
          0
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 7,
        "question": "Is there a couch?",
        "dependency": [
          6
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 8,
        "question": "Is there a coffee table?",
        "dependency": [
          6
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 9,
        "question": "Are there magazines?",
        "dependency": [
          6
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      {
        "proposition_id": 10,
        "question": "Is there a remote control?",
        "dependency": [
          6
        ],
        "category_broad": "entity",
        "category_detailed": "whole"
      },
      "... [TRUNCATED 9 more items] ..."
    ],
    "category": "",
    "tags": [],
    "image_path": ""
  }
}
```

*注意：部分内容为显示目的已截断。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets dpg_bench \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='text2image',
    model_task='image_generation',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['dpg_bench'],
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
