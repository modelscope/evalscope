# JobBench


## 概述

JobBench 评估智能体系统在真实专业工作场景中的表现，这些任务要求系统能够阅读参考文件、生成交付成果，并整合多源信息。本适配器使用 ModelScope 数据集 `evalscope/job-bench`。

## 任务描述

- **任务类型**：智能体专业工作 / 交付成果生成
- **输入**：包含可选参考文件（位于 `reference_files/` 目录下）的职场风格任务提示
- **输出**：最终交付成果文件写入 `jobbench_output/` 目录
- **数据集**：ModelScope `evalscope/job-bench`
- **评估指标**：加权 LLM 评分标准得分（`normalized_score`）

## 评估说明

- 默认评估划分（split）为 `main`。
- 请配置 `judge_model_args` 以进行评分标准打分。
- `normalized_score` 是主要的加权得分（`total_score / max_score`）；`pass_rate` 是完全通过评分项的未加权比例；`total_score` 是通过评分项权重的原始总和。
- Docker 运行默认使用 `python:3.11-slim-bookworm` 镜像。正式评估时，请提供包含任务所需 Office、PDF 和电子表格工具的镜像。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `job_bench` |
| **数据集ID** | [evalscope/job-bench](https://modelscope.cn/datasets/evalscope/job-bench/summary) |
| **论文** | 无 |
| **标签** | `Agent`, `Knowledge`, `MultiTurn` |
| **指标** | `normalized_score`, `pass_rate`, `total_score` |
| **默认示例数** | 0-shot |
| **评估划分** | `main` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 65 |
| 提示词长度（平均） | 3344.32 字符 |
| 提示词长度（最小/最大） | 2000 / 4920 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "203fd1aa",
      "content": "You are preparing the Statistical Analysis Plan for Phase III trial XR-2847 evaluating cardiovascular disease prevention. The sponsor requires validation of statistical assumptions against empirical data and regulatory alignment before protoc ... [TRUNCATED 2026 chars] ... erables.\n\nWrite every final deliverable file under `jobbench_output`. Do not put intermediate scratch files there.\nYour final message may summarize what you produced, but files requested by the task must be actual files in\n`jobbench_output`.\n"
    }
  ],
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
    },
    {
      "name": "python_exec",
      "description": "Execute Python source code inside the sandbox environment. Returns stdout and stderr output.",
      "parameters": {
        "properties": {
          "code": {
            "type": "string",
            "description": "Python source code to execute."
          },
          "timeout": {
            "type": "number",
            "description": "Maximum execution time in seconds (default: 60).",
            "default": 60
          }
        },
        "required": [
          "code"
        ]
      }
    }
  ],
  "metadata": {
    "task_id": "biostatisticians__task1",
    "reference_files": [
      "dataset/biostatisticians/task1/task_folder/Clinical_Study_Proposal_CVD_Prevention.csv",
      "dataset/biostatisticians/task1/task_folder/framingham.csv"
    ],
    "rubric_json": "{\n  \"rubrics\": [\n    {\n      \"rubric\": \"Does the analysis calculate the observed 10-year CHD event rate in the eligible population as 20.2% (236/1166) and identify this as higher than the assumed 15% control group rate?\",\n      \"weight\": 10,\n ... [TRUNCATED 4642 chars] ... ge criterion and by the sysBP criterion separately\",\n        \"The flow shows the final eligible population count (1,166)\",\n        \"The flow enables verification of the filtering process and assessment of generalizability\"\n      ]\n    }\n  ]\n}"
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
    --datasets job_bench \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":250}' \
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
    datasets=['job_bench'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=250,
    ),
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
