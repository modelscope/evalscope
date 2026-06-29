# GDPval


## 概述

GDPval 用于评估模型是否能够完成具有现实经济价值的工作任务，并生成所要求的交付文件。该适配器针对 OpenAI 公开的 220 个任务的 gold 子集，该子集在 ModelScope 上镜像为 `openai-mirror/gdpval`。

## 任务描述

- **任务类型**：代理式专业工作 / 交付物生成
- **输入**：类似工作场景的任务提示，可选附带参考文件
- **输出**：最终回复文本以及位于 `deliverable_files/` 目录下的请求文件
- **数据集**：OpenAI 公开的 GDPval gold 子集，包含 220 个任务

## 主要特性

- 使用原生 EvalScope 的 `AgentLoopAdapter`，支持 bash 和 Python 执行工具。
- 默认从 ModelScope 加载记录和参考文件。
- 将选定的参考文件以只读方式挂载到沙箱中的 `/reference_files` 目录下。
- 在沙箱销毁前提取写入 `deliverable_files/` 的文件。
- 生成包含 `deliverable_text` 和 `deliverable_files` 列的 GDPval 提交包。

## 评估说明

- 默认 Docker 镜像为 `evalscope/gdpval:latest`，当本地缺失时会根据捆绑的 Dockerfile 自动构建。设置 `extra_params.auto_build_docker_image=false` 可强制使用预构建镜像，或通过 `extra_params.docker_image` 覆盖镜像。
- `submission_ready` 是一个本地就绪度指标：当模型生成了最终文本或至少一个交付文件时值为 1。它不是官方的 GDPval 质量评分。
- `llm_rubric_score` 是 EvalScope 实现的非官方 LLM 评判分数，基于任务评分标准、最终文本和交付文件清单计算得出。它不是 OpenAI/GDPval 官方评分。
- 文档/电子表格/幻灯片的完整质量依赖于 GDPval 运行时镜像。轻量级 Python 镜像仅适用于管道连通性冒烟测试。

## 评分与提交

- `scoring_mode=export_only` 会在 EvalScope 报告目录下生成本地提交文件夹。
- `scoring_mode=llm_rubric_judge` 额外运行 EvalScope 自有的非官方 LLM 评分标准评判器。这适用于冒烟测试和回归检查，但不是官方 GDPval 评分。
- EvalScope 不在本地计算官方 GDPval 分数。Inspect Evals 的实现同样导出提交数据集用于外部评分，而非提供本地官方评判器。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `gdpval` |
| **数据集ID** | [openai-mirror/gdpval](https://modelscope.cn/datasets/openai-mirror/gdpval/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `Knowledge`, `MultiTurn` |
| **指标** | `submission_ready`, `llm_rubric_score` |
| **默认示例数** | 0-shot |
| **评估分割** | `train` |


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
      "id": "84e882ee",
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
| `dataset_hub` | `str` | `modelscope` | 用于加载 GDPval 记录和参考文件的数据集中心。选项: ['modelscope', 'huggingface', 'local'] |
| `dataset_revision` | `str` | `` | 可选的数据集版本。留空则使用中心默认版本。 |
| `max_steps` | `int` | `250` | 每个样本的最大代理步骤数。 |
| `command_timeout` | `float` | `180.0` | 每条命令的默认超时时间（秒）。 |
| `docker_image` | `str` | `evalscope/gdpval:latest` | 用作每个样本沙箱的 Docker 镜像。 |
| `auto_build_docker_image` | `bool` | `True` | 如果本地缺少默认 GDPval Docker 镜像，则自动构建。 |
| `network_enabled` | `bool` | `True` | 允许沙箱访问网络。 |
| `download_reference_files` | `bool` | `True` | 在推理前从数据集中心下载每个选定样本的参考文件。 |
| `scoring_mode` | `str` | `export_only` | 评分模式：导出本地提交包或运行 EvalScope 自有的非官方 LLM 评分标准评判器。选项: ['export_only', 'llm_rubric_judge'] |

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

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['gdpval'],
    dataset_args={
        'gdpval': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```