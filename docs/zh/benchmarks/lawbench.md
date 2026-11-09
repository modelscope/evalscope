# LawBench


## 概述

LawBench 通过 20 个任务评估中文法律知识，涵盖记忆、理解和应用三个层次。本原生适配遵循[官方零样本数据和评分代码](https://github.com/open-compass/LawBench)，对应版本为 `e30981bb3ff54c41571f222e0b23e92d27375388`。

## 任务描述

- **任务类型**: 法律问答、分类、抽取、生成和文书校对
- **输入**: 官方任务指令与问题，以换行符连接
- **输出**: 遵循任务要求格式的中文回答
- **领域**: 中文法律知识与法律推理

## 主要特点

- 共 10,000 个样本，每个任务 500 个；子集为 `1-1`–`1-2`、`2-1`–`2-10` 和 `3-1`–`3-8`
- 涵盖法条背诵、法律考试、文书校对、争议分类、阅读理解、实体与事件抽取、摘要、论辩挖掘、判决预测、金额计算和法律咨询
- 保留官方基于子串的答案解析、字符级 F1、ROUGE-L 和归一化刑期对数距离
- 任务 `2-1` 使用官方 ChERRANT 字符级语料 F0.5，对参与评测的记录整体计算一次

## 评估说明

- 使用 `pip install 'evalscope[lawbench]'` 安装普通任务的评分依赖。
- 下载官方仓库，将 `dataset_args.lawbench.local_path` 设为仓库根目录、`data/zero_shot` 目录，或包含 20 个 `<task>.json` 文件的目录。本适配暂不自动下载完整的 Hub 镜像。
- 仅支持零样本评测。可通过 `subset_list` 选择任务，支持框架标准的 `limit`、`shuffle`、过滤器和 `repeats` 设置。
- `lawbench_score` 为任务的官方得分，通常位于 [0, 1]；刑期对数距离得分可能为负数。总分及类别得分均为相应任务得分的等权平均值。
- 官方跳过的争议焦点或刑期参考答案不计入得分分母，但保留在弃权率分母中。NER 上游拼写错误的 `anstention_rate` 统一为 `abstention_rate`；官方 CSV 入口会将该拼写错误字段静默记为零。
- 任务 `2-1` 还需要官方 `evaluation/utils` 目录及其 ChERRANT 依赖（`ltp`、`pypinyin`、`opencc-python-reimplemented`、`timeout-decorator`、`python-Levenshtein`）。将 `extra_params.cherrant_path` 指向该目录，也可通过 `cherrant_python` 指定独立的 Python 解释器。评分在独立临时目录中运行；工具错误和超时会明确报错。
- 无需 LLM 裁判或 OpenCompass 运行环境。本基准为 LawBench，与 PLawBench 不同。
- 下方通用使用示例还需要填写 `local_path`；选择任务 `2-1` 时，还需填写 `cherrant_path`。完整配置如下：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-model', eval_type='openai_api',
    api_url='http://localhost:8000/v1', api_key='EMPTY',
    datasets=['lawbench'],
    dataset_args={'lawbench': {
        'local_path': '/path/to/LawBench',
        'extra_params': {'cherrant_path': '/path/to/LawBench/evaluation/utils'},
    }},
    generation_config={'temperature': 0, 'max_tokens': 1024},
))
```


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `lawbench` |
| **数据集 ID** | `LawBench` |
| **论文** | [论文](https://arxiv.org/abs/2309.16289) |
| **标签** | `Knowledge`, `QA`, `Reasoning` |
| **指标** | `lawbench_score`, `abstention_rate` |
| **默认示例数** | 零样本 |
| **评测划分** | `zero_shot` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 10,000 |
| 提示词长度（平均） | 391.46 字符 |
| 提示词长度（最小/最大） | 32 / 23490 字符 |

**各子集统计:**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
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
      "id": "80213160",
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

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `cherrant_path` | `str` | `` | 官方 LawBench 的 evaluation/utils 目录；任务 2-1 必填。 |
| `cherrant_python` | `str` | `` | 已安装 ChERRANT 依赖的 Python 可执行文件，默认使用当前解释器。 |
| `cherrant_timeout` | `float` | `600.0` | 每个 ChERRANT 子进程的超时时间，单位为秒。 |

## 使用方法

### 使用命令行

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
