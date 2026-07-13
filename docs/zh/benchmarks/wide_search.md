# WideSearch


## 概述

WideSearch 用于评估搜索智能体在广泛网络信息检索任务上的表现。每个任务要求智能体收集大量原子事实，并返回一个结构化的 Markdown 表格。EvalScope 使用 ModelScope 上的 `bytedance-community/WideSearch` 数据集。

## 任务描述

- **任务类型**：多轮搜索智能体
- **输入**：包含明确表格结构的自然语言信息收集请求
- **输出**：完整的 Markdown 表格
- **数据集**：`full` 划分中共有 200 个任务；其中 100 个为英文，100 个为中文

## 核心特性

- 官方单智能体协议：支持语言特定的系统提示词、``function_calling``，以及默认 50 步的最大执行步数。
- 默认为每个样本提供临时本地目录中的 Bash 环境；Docker 沙箱和 MCP 服务器为可选项。
- 单次完整运行即可生成 ``all``、``en`` 和 ``zh`` 三份报告，无需重复推理。

## 评估说明

- 采用官方的 Markdown 表格对齐方式及混合规则/LLM 评分语义。
- 必须设置 ``judge_strategy='auto'`` 或 ``'llm'`` 并显式指定 ``judge_model_args``；不支持仅使用规则评分。
- 运行示例及论文风格的重复实验设置，请参阅 [WideSearch 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/third_party/wide_search.html)。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `wide_search` |
| **数据集ID** | [bytedance-community/WideSearch](https://modelscope.cn/datasets/bytedance-community/WideSearch/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2508.07999) |
| **标签** | `Agent`, `MultiTurn`, `Retrieval` |
| **指标** | `success_rate`, `row_precision`, `row_recall`, `row_f1`, `item_precision`, `item_recall`, `item_f1` |
| **默认示例数** | 0-shot |
| **评估划分** | `full` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 200 |
| 提示词长度（平均） | 631.01 字符 |
| 提示词长度（最小/最大） | 182 / 1807 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "59f736f6",
      "content": "My son is about to start his university applications in 2025 for postgraduates but he’s still uncertain about both his major and which universities to apply to. Could you help me find the top five universities in each of the five broad subjec ... [TRUNCATED 691 chars] ... names in English. \nUse only Arabic numerals in the ranking, for example: 1.\nDon't ask me any questions, just output the results according to the columns without omitting cells arbitrarily. The output format is \n```markdown\n{data_content}\n```."
    }
  ],
  "target": "Subject,University,Country,QS World University Rankings by Subject 2025,QS World University Rankings 2025,Times Higher Education  World University Rankings 2025,Home Page,Application Deadline,Application Fee\nArts & Humanities,Harvard Universi ... [TRUNCATED 2837 chars] ... nuary 5,$90\nSocial Sciences & Management,Massachusetts Institute of Technology,United States,4,1,2,https://www.mit.edu/,January 6 ,$75\nSocial Sciences & Management,University of Cambridge,United Kingdom,5,5,5,https://www.cam.ac.uk/,Oct 15,£60",
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "bash",
      "description": "Execute a bash command inside the sandbox environment. Returns the combined stdout / stderr output of the command.",
      "parameters": {
        "properties": {
          "command": {
            "type": "string",
            "description": "The bash command to execute."
          },
          "timeout": {
            "type": "number",
            "description": "Maximum execution time in seconds (default: 60).",
            "default": 60
          }
        },
        "required": [
          "command"
        ]
      }
    }
  ],
  "metadata": {
    "instance_id": "ws_en_001",
    "language": "en",
    "evaluation": {
      "unique_columns": [
        "subject",
        "university"
      ],
      "required": [
        "subject",
        "university",
        "qsworlduniversityrankingsbysubject2025",
        "qsworlduniversityrankings2025",
        "timeshighereducationworlduniversityrankings2025",
        "homepage",
        "applicationdeadline",
        "applicationfee"
      ],
      "eval_pipeline": {
        "applicationdeadline": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "llm_judge"
          ],
          "criterion": "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence.\nThe month and day must be correct"
        },
        "applicationfee": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "llm_judge"
          ],
          "criterion": "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence.\nIf there are multiple fees in the reference answer, all must be included."
        },
        "homepage": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "url_match"
          ]
        },
        "subject": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "university": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "qsworlduniversityrankingsbysubject2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "qsworlduniversityrankings2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "timeshighereducationworlduniversityrankings2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        }
      }
    }
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
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
    --datasets wide_search \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":50}' \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['wide_search'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=50,
    ),
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
