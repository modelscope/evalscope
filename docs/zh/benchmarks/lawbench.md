# LawBench


## 概述

LawBench 通过涵盖记忆、理解和应用的 20 个任务来评估中文法律知识。此原生适配器遵循 [官方零样本数据和评分代码](https://github.com/open-compass/LawBench)，基于修订版本 `e30981bb3ff54c41571f222e0b23e92d27375388`。

## 任务描述

- **任务类型**：法律问答、分类、抽取、生成和校对
- **输入**：官方指令后跟问题，两者之间用换行符分隔
- **输出**：符合任务要求格式的中文自由文本响应
- **领域**：中文法律知识和法律推理

## 主要特点

- 10,000 个样本：每个任务 500 个；子集包括 `1-1`–`1-2`、`2-1`–`2-10` 和 `3-1`–`3-8`
- 涵盖法条背诵、法律考试、校对、争议焦点分类、阅读理解、实体/事件抽取、摘要、论辩挖掘、判决预测、损害赔偿计算和咨询
- 保留官方的基于子串的解析答案方式、字符 F1、ROUGE-L 以及归一化的刑期对数距离
- 任务 `2-1` 使用官方的 ChERRANT 字符级语料库 F0.5，该指标是对全部参与评测的记录整体计算一次

## 评估说明

- 安装常规评分依赖项：`pip install 'evalscope[lawbench]'`。
- 20 个官方零样本任务文件会自动从 [ModelScope](https://modelscope.cn/datasets/evalscope/LawBench) 加载。每个文件均为固定官方修订版本的未修改副本。
- 对于离线数据，请将 `dataset_args.lawbench.local_path` 设置为官方检出根目录、其 `data/zero_shot` 目录或包含这 20 个 `<task>.json` 文件的另一个目录。
- 仅支持零样本。可通过 `subset_list` 选择单个任务；标准的 `limit`、`shuffle`、过滤器和 `repeats` 参数均适用。
- `lawbench_score` 是任务的官方得分，通常在 [0, 1] 范围内；刑期对数距离得分可能为负值。总体得分和类别得分为本次参与评测的任务得分的未加权平均值。
- 官方跳过的争议焦点/刑期参考答案不计入得分分母，但保留在弃答率分母中。NER 上游字段拼写错误 `anstention_rate` 已规范化为 `abstention_rate`；官方 CSV 驱动程序则会将此拼写错误的字段静默默认设为零。
- 任务 `2-1` 额外需要官方的 `evaluation/utils` 目录及其 ChERRANT 依赖项（`ltp`、`pypinyin`、`opencc-python-reimplemented`、`timeout-decorator`、`python-Levenshtein`）。将 `extra_params.cherrant_path` 设置为该目录，并可选地将 `cherrant_python` 设置为单独的 Python 解释器。评分在私有临时目录中运行；工具错误和超时会导致显式失败。
- 不需要 LLM 裁判或 OpenCompass 运行环境。这是 LawBench，与 PLawBench 不同。
- 当选择任务 `2-1`（默认包含）时，请在下方的通用使用方法示例中添加 `cherrant_path`。完整配置如下：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-model', eval_type='openai_api',
    api_url='http://localhost:8000/v1', api_key='EMPTY',
    datasets=['lawbench'],
    dataset_args={'lawbench': {
        'extra_params': {'cherrant_path': '/path/to/LawBench/evaluation/utils'},
    }},
    generation_config={'temperature': 0, 'max_tokens': 1024},
))
```


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `lawbench` |
| **数据集ID** | [evalscope/LawBench](https://modelscope.cn/datasets/evalscope/LawBench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2309.16289) |
| **标签** | `Knowledge`, `QA`, `Reasoning` |
| **指标** | `lawbench_score`, `abstention_rate` |
| **默认示例数** | 0-shot |
| **评估划分** | `zero_shot` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 10,000 |
| 提示词长度（平均） | 391.46 字符 |
| 提示词长度（最小/最大） | 32 / 23490 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `1-1` | 500 | 40.74 | 32 | 55 |
| `1-2` | 500 | 219.41 | 106 | 503 |
| `2-1` | 500 | 154.61 | 95 | 527 |
| `2-2` | 500 | 823.05 | 316 | 4524 |
| `2-3` | 500 | 276.44 | 225 | 498 |
| `2-4` | 500 | 213.21 | 181 | 357 |
| `2-5` | 500 | 756.01 | 455 | 1441 |
| `2-6` | 500 | 151.76 | 95 | 375 |
| `2-7` | 500 | 350.63 | 69 | 396 |
| `2-8` | 500 | 484.33 | 282 | 1045 |
| `2-9` | 500 | 215.31 | 167 | 474 |
| `2-10` | 500 | 157.27 | 98 | 377 |
| `3-1` | 500 | 679.97 | 127 | 22218 |
| `3-2` | 500 | 113.21 | 73 | 196 |
| `3-3` | 500 | 585.15 | 103 | 9861 |
| `3-4` | 500 | 684.82 | 146 | 22241 |
| `3-5` | 500 | 1007.58 | 243 | 23490 |
| `3-6` | 500 | 293.43 | 135 | 939 |
| `3-7` | 500 | 535.58 | 136 | 2053 |
| `3-8` | 500 | 86.62 | 79 | 129 |

## 样例示例

**子集**: `1-1`

```json
{
  "input": [
    {
      "id": "ac43c78e",
      "content": "回答以下问题，只需直接给出法条内容:\n民法商法农民专业合作社法第三十三条的内容是什么？"
    }
  ],
  "target": "答案:农民专业合作社设理事长一名，可以设理事会。理事长为本社的法定代表人。,农民专业合作社可以设执行监事或者监事会。理事长、理事、经理和财务会计人员不得兼任监事。,理事长、理事、执行监事或者监事会成员，由成员大会从本社成员中选举产生，依照本法和章程的规定行使职权，对成员大会负责。,理事会会议、监事会会议的表决，实行一人一票。",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task": "1-1"
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
| `cherrant_path` | `str` | `` | 官方 LawBench evaluation/utils 目录；任务 2-1 必需。 |
| `cherrant_python` | `str` | `` | 包含 ChERRANT 依赖项的 Python 可执行文件；默认为当前解释器。 |
| `cherrant_timeout` | `float` | `600.0` | 每个 ChERRANT 子进程的超时时间（秒）。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets lawbench \
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
    datasets=['lawbench'],
    dataset_args={
        'lawbench': {
            # subset_list: ['1-1', '1-2', '2-1']  # 可选，用于评估特定子集
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
