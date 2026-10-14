# SPaRC

## 概述

SPaRC（空间路径推理挑战）基于游戏《见证者》测试二维网格谜题中的逐步空间推理。模型必须从起始节点到结束节点绘制一条线，这条线必须遵守网格上的所有规则。

## 任务描述

- **任务类型**：在规则约束下的空间路径寻找
- **输入**：一个文本网格（起始和结束节点、空格、点、彩色方块、星星、三角形、多连块）以及游戏规则
- **输出**：逐步推理，然后是标记 `####` 和路径作为 `(x, y)` 节点坐标的列表
- **领域**：空间推理、约束满足

## 主要特点

- 500 个测试谜题，分为五个难度级别（级别 1 到级别 5：86 / 118 / 121 / 86 / 89 个谜题）；论文中称级别 5 为“困难”
- 每个谜题的所有有效解决方案都存储在数据集中，因此评分不需要裁判模型
- 在论文中，人类解决了 98.0% 的测试谜题（级别 5 中的 94.5%），而最好的模型 o4-mini 解决了 15.8%（级别 5 中的 1.1%）
- 提示词是论文中的单次提示，提示词中包含一个示例

## 评估说明

- 子集是难度级别；`subset_list: ['Level 5']` 仅评估困难谜题。`limit` 适用于每个级别，因此 `limit=4` 会提供 20 个谜题（每个级别 4 个）
- `accuracy`（主要指标）：最后一个 `####` 后的路径必须与谜题存储的解决方案之一完全匹配
- `invalid_path_rate`（越低越好）：答案中不是从起点到终点的有效连线的比例（错误的起点或终点、步骤之间的空隙、重复访问的节点或在规则单元格上的步骤）。这是论文中表 7 的“Invalid Path”列
- 提示生成器和验证器是 `sparc-puzzle` 包中参考代码的副本；它们在所有 500 个测试谜题上提供相同的提示和相同的分数
- 数据集是 ModelScope 上的 `evalscope/SPaRC`（配置 `all`），为了可复现性，版本已固定
- 推理模型通常每道谜题会写超过 10k 个 token，因此请设置较大的 `max_tokens`
- [论文](https://aclanthology.org/2025.emnlp-main.526/) | [代码](https://github.com/lkaesberg/SPaRC) | [数据集](https://www.modelscope.cn/datasets/evalscope/SPaRC) | [网站](https://sparc.gipplab.org/)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `sparc` |
| **数据集 ID** | [evalscope/SPaRC](https://modelscope.cn/datasets/evalscope/SPaRC/summary) |
| **论文** | [论文](https://arxiv.org/abs/2505.16686) |
| **标签** | `Reasoning` |
| **指标** | `accuracy`, `invalid_path_rate` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 500 |
| 提示词长度（平均值） | 7753.07 字符 |
| 提示词长度（最小值/最大值） | 7478 / 8589 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Level 1` | 86 | 7565.24 | 7478 | 7746 |
| `Level 2` | 118 | 7668.82 | 7484 | 7963 |
| `Level 3` | 121 | 7715.45 | 7486 | 8018 |
| `Level 4` | 86 | 7848.93 | 7594 | 8184 |
| `Level 5` | 89 | 8004.79 | 7610 | 8589 |

## 样例示例

**子集**: `Level 1`

```json
{
  "input": [
    {
      "id": "a7ac7618",
      "content": "You are an expert at solving puzzles games."
    },
    {
      "id": "468c49a5",
      "content": "\n    ## Objective\n    You are a specialized AI proficient in spatial reasoning and solving puzzles from the game 'The Witness'. Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provide ... [TRUNCATED 7215 chars] ...  marker string `####` followed immediately by the solution path as a list of node coordinates `(x, y)`. Include all intermediate nodes from start to end.\n\n    **Example Solution Path Format:**\n    ####\n    (0,0), (1,0), (2,0), (2,1), ...\n    "
    }
  ],
  "target": "(5,0), (6,0), (6,1), (6,2), (5,2), (4,2), (4,3), (4,4), (5,4), (6,4), (6,5), (6,6), (5,6), (4,6), (3,6), (2,6), (1,6), (0,6), (0,5), (0,4), (1,4), (2,4), (2,3), (2,2), (1,2), (0,2), (0,1), (0,0), (1,0), (2,0), (3,0), (4,0)",
  "id": 0,
  "group_id": 0,
  "subset_key": "Level 1",
  "metadata": {
    "difficulty_level": 1,
    "difficulty_score": 0.842447049854198,
    "solution_count": 3,
    "puzzle": {
      "id": "3fe8a85b768ae5b7",
      "grid_size": {
        "height": 3,
        "width": 4
      },
      "puzzle_array": [
        [
          "+",
          ".",
          "+",
          "+",
          "E",
          "S",
          "+",
          "+",
          "+"
        ],
        [
          "+",
          "N",
          "+",
          "N",
          "+",
          "N",
          "+",
          "P-B-1824",
          "+"
        ],
        [
          "+",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+"
        ],
        [
          "+",
          "N",
          "+",
          "N",
          "+",
          "N",
          "+",
          "N",
          "G"
        ],
        [
          ".",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+"
        ],
        [
          "+",
          "N",
          "+",
          "N",
          "+",
          "N",
          "+",
          "N",
          "+"
        ],
        [
          "+",
          "+",
          "+",
          "+",
          "+",
          ".",
          "+",
          "+",
          "+"
        ]
      ],
      "polyshapes": "{\"1824\":[[0,0,1,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]]}",
      "solutions": [
        {
          "index": 0,
          "path": [
            {
              "x": 5,
              "y": 0
            },
            {
              "x": 6,
              "y": 0
            },
            {
              "x": 6,
              "y": 1
            },
            {
              "x": 6,
              "y": 2
            },
            {
              "x": 5,
              "y": 2
            },
            {
              "x": 4,
              "y": 2
            },
            {
              "x": 4,
              "y": 3
            },
            {
              "x": 4,
              "y": 4
            },
            {
              "x": 5,
              "y": 4
            },
            {
              "x": 6,
              "y": 4
            },
            "... [TRUNCATED 22 more items] ..."
          ],
          "pathLength": 32
        },
        {
          "index": 1,
          "path": [
            {
              "x": 5,
              "y": 0
            },
            {
              "x": 6,
              "y": 0
            },
            {
              "x": 6,
              "y": 1
            },
            {
              "x": 6,
              "y": 2
            },
            {
              "x": 5,
              "y": 2
            },
            {
              "x": 4,
              "y": 2
            },
            {
              "x": 4,
              "y": 3
            },
            {
              "x": 4,
              "y": 4
            },
            {
              "x": 5,
              "y": 4
            },
            {
              "x": 6,
              "y": 4
            },
            "... [TRUNCATED 18 more items] ..."
          ],
          "pathLength": 28
        },
        {
          "index": 2,
          "path": [
            {
              "x": 5,
              "y": 0
            },
            {
              "x": 6,
              "y": 0
            },
            {
              "x": 6,
              "y": 1
            },
            {
              "x": 6,
              "y": 2
            },
            {
              "x": 5,
              "y": 2
            },
            {
              "x": 4,
              "y": 2
            },
            {
              "x": 4,
              "y": 3
            },
            {
              "x": 4,
              "y": 4
            },
            {
              "x": 5,
              "y": 4
            },
            {
              "x": 6,
              "y": 4
            },
            "... [TRUNCATED 18 more items] ..."
          ],
          "pathLength": 28
        }
      ]
    }
  }
}
```

*注：部分内容因显示原因被截断。*

## 提示模板

**系统提示:**
```text
You are an expert at solving puzzles games.
```

**提示模板:**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets sparc \
    --limit 10  # Remove this line for formal evaluation
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['sparc'],
    dataset_args={
        'sparc': {
            # subset_list: ['Level 1', 'Level 2', 'Level 3']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
