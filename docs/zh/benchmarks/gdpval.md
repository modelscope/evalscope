# GDPval


## 概述

GDPval 评估模型是否能够完成具有现实经济价值的工作任务，并生成所要求的交付文件。该适配器针对 OpenAI 公开的 220 个任务的 gold 子集，该子集在 ModelScope 上镜像为 `openai-mirror/gdpval`。

## 任务描述

- **任务类型**：代理式专业工作 / 交付物生成
- **输入**：职场风格的任务提示，可选附带参考文件
- **输出**：最终回复文本及位于 `deliverable_files/` 目录下的请求文件
- **数据集**：OpenAI 公开的 GDPval gold 子集，包含 220 个任务

## 主要特性

- 使用原生 EvalScope 的 `AgentLoopAdapter`，支持 bash 和 Python 执行工具。
- 默认从 ModelScope 加载记录和参考文件。
- 将选定的参考文件以只读方式挂载到沙箱中的 `/reference_files` 目录下。
- 在沙箱销毁前提取写入 `deliverable_files/` 的文件。
- 生成符合 GDPval 提交格式的包，包含 `deliverable_text` 和 `deliverable_files` 列。

## 评估说明

- 默认 Docker 镜像是根据捆绑的 Dockerfile 自动构建的，并打上基于内容哈希的本地标签。设置
  `extra_params.auto_build_docker_image=false` 可强制使用预构建的 `evalscope/gdpval:latest` 镜像。可通过 `TaskConfig.sandbox.default_config` 覆盖镜像、网络、CPU 或内存配置。
- `submission_ready` 是一个本地就绪指标：当模型生成了最终文本或至少一个交付文件时，其值为 1。它不是官方的 GDPval 质量评分。
- EvalScope 不运行本地的 GDPval 评判器。请使用导出的提交包配合 OpenAI 官方的 GDPval 评判器获取质量评分。
- 完整的文档/电子表格/幻灯片质量依赖于 GDPval 运行时镜像。轻量级 Python 镜像仅适用于管道连通性冒烟测试。

## 评分与提交

- EvalScope 在报告目录下写入一个本地提交文件夹。
- 提交内容包含符合 GDPval 数据集格式的 `deliverable_text` 和 `deliverable_files` 字段。
- 官方 GDPval 评分是外部进行的。请在导出的提交包上运行 OpenAI 官方的 GDPval 评判器。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `gdpval` |
| **数据集ID** | [openai-mirror/gdpval](https://modelscope.cn/datasets/openai-mirror/gdpval/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `Knowledge`, `MultiTurn` |
| **指标** | `submission_ready` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 220 |
| 提示词长度（平均） | 2742.59 字符 |
| 提示词长度（最小/最大） | 1058 / 7160 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "3315415d",
      "content": "You are an auditor and as part of an audit engagement, you are tasked with reviewing and testing the accuracy of reported Anti-Financial Crime Risk Metrics.\n\nThe attached spreadsheet titled ‘Population’ contains Anti-Financial Crime Risk Metr ... [TRUNCATED 2069 chars] ...  folder named `deliverable_files` in the sandbox working directory.\nWe will grade your final message as part of the deliverable, but requested documents, spreadsheets, slides, media,\nor archives should be actual files in `deliverable_files`.\n"
    }
  ],
  "target": "",
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
    "task_id": "83d10b06-26d1-4636-a32c-23f92c57f30b",
    "sector": "Professional, Scientific, and Technical Services",
    "occupation": "Accountants and Auditors",
    "prompt": "You are an auditor and as part of an audit engagement, you are tasked with reviewing and testing the accuracy of reported Anti-Financial Crime Risk Metrics.\n\nThe attached spreadsheet titled ‘Population’ contains Anti-Financial Crime Risk Metr ... [TRUNCATED 1526 chars] ... across all Divisions and sub-Divisions.\n\n4. Create a new spreadsheet titled ‘Sample’:\n- Tab 1: Selected sample, copied from the original ‘Population’ sheet, with selected rows marked in column K.\n- Tab 2: Workings for sample size calculation.",
    "reference_files": [
      "reference_files/cc781e4dc0985c8eb327a53ec03b5900/Population v2.xlsx"
    ],
    "reference_file_urls": [
      "https://huggingface.co/datasets/openai/gdpval/resolve/main/reference_files/cc781e4dc0985c8eb327a53ec03b5900/Population%20v2.xlsx"
    ],
    "reference_file_hf_uris": [
      "hf://datasets/openai/gdpval@main/reference_files/cc781e4dc0985c8eb327a53ec03b5900/Population%20v2.xlsx"
    ],
    "reference_paths": [
      "reference_files/Population v2.xlsx"
    ],
    "sandbox_reference_paths": [
      "/reference_files/cc781e4dc0985c8eb327a53ec03b5900/Population v2.xlsx"
    ],
    "rubric_pretty": "[+2] The submitted deliverable is an Excel workbook file whose basename is 'Sample' (accept .xlsx, .xls, or .xlsm).\n\n[+2] The workbook contains a worksheet named exactly 'Sample Size Calculation' (case-insensitive, ignoring surrounding spaces ... [TRUNCATED 4861 chars] ... ntage changes (e.g., |J| ≥ 100%) are made easily identifiable (such as by a separate flag, note, or conditional formatting).\n\n[+1] The first worksheet is named 'Sample' (case-insensitive).\n\n[+5] Overall formatting and style of the deliverable",
    "rubric_json": "[{\"score\": 2, \"criterion\": \"The submitted deliverable is an Excel workbook file whose basename is 'Sample' (accept .xlsx, .xls, or .xlsm).\", \"required\": null, \"rubric_item_id\": \"1d43f1eb-4011-47ac-8ad7-a3c467639a6a\", \"author_type\": \"human\", \" ... [TRUNCATED 11817 chars] ... ull}, {\"score\": 5, \"criterion\": \"Overall formatting and style of the deliverable\", \"required\": null, \"rubric_item_id\": \"a64588ed-db04-4b8b-b3b8-3674ddcf10d1\", \"author_type\": \"human\", \"tags\": [\"true\"], \"read_only\": null, \"form_content\": null}]",
    "dataset_id": "openai-mirror/gdpval",
    "dataset_hub": "modelscope"
  }
}
```

## 提示模板

**提示模板:**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `auto_build_docker_image` | `bool` | `True` | 如果本地缺少默认 GDPval Docker 镜像，则自动构建。 |
| `download_reference_files` | `bool` | `True` | 在推理前从数据集中心下载每个选定样本的参考文件。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gdpval \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['gdpval'],
    # agent_config=NativeAgentConfig(
    #     strategy='function_calling',
    #     max_steps=250,
    # ),
    dataset_args={
        'gdpval': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
