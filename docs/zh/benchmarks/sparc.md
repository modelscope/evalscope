# SPaRC


## 概述

SPaRC（空间路径规划推理挑战）基于游戏《The Witness》中的二维网格谜题，测试模型的逐步空间推理能力。模型需要绘制一条从起点到终点的路径，且必须遵守网格上的所有规则。

## 任务描述

- **任务类型**：在规则约束下的空间路径规划
- **输入**：文本网格（包含起点和终点节点、空隙、圆点、彩色方块、星形、三角形、多联骨牌）以及游戏规则
- **输出**：逐步推理过程，随后是标记 `####` 和路径坐标列表 `(x, y)` 的节点序列
- **领域**：空间推理、约束满足

## 主要特点

- 包含500个测试谜题，分为五个难度等级（第1级到第5级：86 / 118 / 121 / 86 / 89个谜题）；论文中将第5级称为"困难"
- 数据集中存储了每个谜题的所有有效解，因此评分无需使用评判模型
- 在论文中，人类解决了98.0%的测试谜题（第5级为94.5%），而最佳模型o4-mini仅解决了15.8%（第5级为1.1%）
- 提示模板采用论文中的单样本提示，并在提示内包含一个示例解答

## 评估说明

- 子集对应难度等级；`subset_list: ['Level 5']` 仅评估困难谜题。`limit` 参数按等级生效，因此 `limit=4` 表示共20个谜题（每个等级4个）
- `accuracy`（主要指标）：最后一个 `####` 之后的路径必须与谜题存储的某个解完全匹配
- `invalid_path_rate`（越低越好）：无效答案的比例，即路径不是从起点到终点的有效连线（起点或终点错误、步骤间存在间隙、重复访问节点、或踩到规则单元格）。对应论文中表7的"Invalid Path"列
- 提示构建器和验证器均来自 `sparc-puzzle` 包中的参考代码，对全部500个测试谜题生成相同的提示和相同的评分
- 数据集从 Hugging Face 加载（`lkaesberg/SPaRC`，配置 `all`，固定版本），目前尚无 ModelScope 副本。默认的 `dataset_hub='modelscope'` 会回退到 Hugging Face
- 推理模型每个谜题通常需要超过10k个token，因此需设置较大的 `max_tokens`
- [论文](https://aclanthology.org/2025.emnlp-main.526/) | [代码](https://github.com/lkaesberg/SPaRC) | [数据集](https://huggingface.co/datasets/lkaesberg/SPaRC) | [网站](https://sparc.gipplab.org/)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `sparc` |
| **数据集ID** | [lkaesberg/SPaRC](https://modelscope.cn/datasets/lkaesberg/SPaRC/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2505.16686) |
| **标签** | `Reasoning` |
| **指标** | `accuracy`, `invalid_path_rate` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 500 |
| 提示词长度（平均） | 7753.07 字符 |
| 提示词长度（最小/最大） | 7478 / 8589 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Level 1` | 86 | 7565.24 | 7478 | 7746 |
| `Level 2` | 118 | 7668.82 | 7484 | 7963 |
| `Level 3` | 121 | 7715.45 | 7486 | 8018 |
| `Level 4` | 86 | 7848.93 | 7594 | 8184 |
| `Level 5` | 89 | 8004.79 | 7610 | 8589 |

## 样例示例

**子集**：`Level 1`

```json
{
  "input": [
    {
      "id": "e9d45478",
      "content": "You are an expert at solving puzzles games."
    },
    {
      "id": "f58f3740",
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

*注：部分内容因显示需要被截断。*

## 提示模板

**系统提示：**
```text
You are an expert at solving puzzles games.
```

**提示模板：**
```text
{question}
```

## 使用方法

### 使用命令行（CLI）

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
