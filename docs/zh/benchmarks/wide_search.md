# WideSearch


## 概述

WideSearch 用于评估搜索智能体在广泛信息检索任务上的表现，这类任务要求从网络中收集大量原子事实，并将其组织成结构化的 Markdown 表格。该基准包含 200 个经人工精心整理的任务，其中英文和中文任务各占一半。

## 任务描述

- **任务类型**：多轮搜索智能体
- **输入**：包含明确表格结构的自然语言信息收集请求
- **输出**：一个完整的 Markdown 表格
- **数据集**：``full`` 划分中共有 200 个任务；其中 100 个为英文，100 个为中文

## 主要特性

- 官方提供的单智能体提示模板，支持 ``function_calling``，默认最多执行 50 步
- 默认在临时本地目录中提供 Bash 工具；可选 Docker 沙箱和 MCP 工具
- 官方实现的 Markdown 解析、表格对齐、预处理，以及混合规则/大语言模型（LLM）评分机制
- 单次完整运行即可同时报告 ``all``、``en`` 和 ``zh`` 的结果，无需重复推理

## 评估说明

- 需设置 ``judge_strategy='auto'`` 或 ``'llm'``，并显式指定 ``judge_model_args``；不支持仅使用规则评分。
- 报告成功率，以及基于 Avg@N、Pass@N 和 Max@N 聚合方式计算的行级/项级精确率、召回率和 F1 分数。
- 通过 ``pip install evalscope[wide_search]`` 安装；若使用 Docker 模式，还需额外安装 ``evalscope[sandbox]``。
- [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/wide_search.html)


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
      "id": "e1d4fcda",
      "content": "My son is about to start his university applications in 2025 for postgraduates but he’s still uncertain about both his major and which universities to apply to. Could you help me find the top five universities in each of the five broad subjec ... [TRUNCATED 691 chars] ... names in English. \nUse only Arabic numerals in the ranking, for example: 1.\nDon't ask me any questions, just output the results according to the columns without omitting cells arbitrarily. The output format is \n```markdown\n{data_content}\n```."
    }
  ],
  "target": "﻿Subject,University,Country,QS World University Rankings by Subject 2025,QS World University Rankings 2025,Times Higher Education  World University Rankings 2025,Home Page,Application Deadline,Application Fee\nArts & Humanities,Harvard Univers ... [TRUNCATED 2838 chars] ... nuary 5,$90\nSocial Sciences & Management,Massachusetts Institute of Technology,United States,4,1,2,https://www.mit.edu/,January 6 ,$75\nSocial Sciences & Management,University of Cambridge,United Kingdom,5,5,5,https://www.cam.ac.uk/,Oct 15,£60",
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
