# ZebraLogicBench

## 概述

ZebraLogicBench 是一个全面的评估框架，用于评估大语言模型（LLM）在源自约束满足问题（CSPs）的逻辑网格谜题上的推理能力。该基准测试重点考察系统性的逻辑推理能力。

## 任务描述

- **任务类型**：逻辑网格谜题求解
- **输入**：包含房屋、属性和线索的逻辑谜题
- **输出**：包含推理过程解释的 JSON 格式解答
- **领域**：约束满足、逻辑演绎

## 主要特点

- 谜题源自约束满足问题（CSP）
- 要求进行系统性的逐步逻辑推理
- 难度等级多样（简单/困难）且规模各异（小型/中型/大型/XL）
- 测试模型处理多个相互依赖线索的能力
- 解答必须为有效的 JSON 格式

## 评估说明

- 默认评估使用 **test** 数据集划分，并采用 **零样本（zero-shot）** 设置
- 跟踪多项指标：
  - `puzzle_acc`：完整正确解答的谜题比例
  - `cell_acc`：正确识别的单个单元格比例
  - 按难度划分：`easy_puzzle_acc`、`hard_puzzle_acc`
  - 按规模划分：`small_puzzle_acc`、`medium_puzzle_acc`、`large_puzzle_acc`、`xl_puzzle_acc`
  - `avg_reason_lens`：平均推理长度
- 输出必须包含推理过程和解答，且格式为 JSON

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `zebralogicbench` |
| **数据集ID** | [allenai/ZebraLogicBench-private](https://modelscope.cn/datasets/allenai/ZebraLogicBench-private/summary) |
| **论文** | N/A |
| **标签** | `Reasoning` |
| **指标** | `puzzle_acc`, `cell_acc`, `easy_puzzle_acc`, `hard_puzzle_acc`, `small_puzzle_acc`, `medium_puzzle_acc`, `large_puzzle_acc`, `xl_puzzle_acc`, `avg_reason_lens`, `no_answer_num` |
| **默认样本数** | 0-shot |
| **评估数据划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,000 |
| 提示词长度（平均） | 3262.38 字符 |
| 提示词长度（最小/最大） | 2011 / 5658 字符 |

## 样例示例

**子集**: `grid_mode`

```json
{
  "input": [
    {
      "id": "e6c901c7",
      "content": "# Example Puzzle\n\nThere are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:\n - Each perso ... [TRUNCATED] ... Animal\": \"___\"\n        },\n        \"House 5\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        }\n    }\n}\n\n"
    }
  ],
  "target": "{\"header\": [\"House\", \"Name\", \"Nationality\", \"BookGenre\", \"Food\", \"Color\", \"Animal\"], \"rows\": [[\"1\", \"Bob\", \"german\", \"mystery\", \"grilled cheese\", \"yellow\", \"dog\"], [\"2\", \"Eric\", \"norwegian\", \"fantasy\", \"stew\", \"blue\", \"fish\"], [\"3\", \"Peter\", \"dane\", \"science fiction\", \"spaghetti\", \"green\", \"cat\"], [\"4\", \"Arnold\", \"swede\", \"biography\", \"stir fry\", \"red\", \"bird\"], [\"5\", \"Alice\", \"brit\", \"romance\", \"pizza\", \"white\", \"horse\"]]}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "created_at": "2024-07-03T21:21:29.209499"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {{
        "House 1": {{
            "Name": "Arnold",
            "Drink": "tea"
        }},
        "House 2": {{
            "Name": "Peter",
            "Drink": "water"
        }},
        "House 3": {{
            "Name": "Eric",
            "Drink": "milk"
        }}
    }}
}}

# Puzzle to Solve

{question}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}


```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets zebralogicbench \
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
    datasets=['zebralogicbench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```