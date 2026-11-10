# LawBench


## 概述

LawBench 通过涵盖记忆、理解与应用能力的 20 项任务，评估中文法律知识及推理能力。其内容包括法律考试、案例分析、信息抽取、文档校对及法律咨询。有关该基准测试及论文的详细信息，请参阅 [官方项目](https://github.com/open-compass/LawBench)。

## 任务描述

- **任务类型**：法律问答、分类、抽取、生成与校对
- **输入**：包含任务指令的中文法律问题或文档
- **输出**：按指定格式生成的中文回答
- **领域**：中文法律知识与法律推理

## 主要特点

- 共 10,000 个样本，每项任务 500 个；可通过子集 `1-1`–`1-2`、`2-1`–`2-10` 和 `3-1`–`3-8` 选择任务
- 任务组覆盖法律知识、文档理解及实际应用
- 数据自动从 [ModelScope](https://modelscope.cn/datasets/evalscope/LawBench) 下载
- 使用官方的特定任务评分规则；无需 LLM 评判器

## 评估说明

- 支持零样本（zero-shot）评估。设置 `subset_list` 以评估选定任务；使用 `limit` 进行简短试用。
- `lawbench_score` 遵循每项任务的官方指标。分数越高越好；大多数分数在 [0, 1] 范围内，而刑期预测分数可能为负值。总体分数和类别分数对每个被评估的任务赋予相等权重。
- `abstention_rate` 报告任务答案解析器未找到有效答案的频率。官方排除的参考答案不计入分数。
- 此基准测试不同于 PLawBench，且不需要 OpenCompass。
- 对于离线数据，请将 `local_path` 设置为官方检出目录、其 `data/zero_shot` 目录，或包含原始 `<task>.json` 文件的目录。

### 安装并运行常规任务

安装常规评分依赖项：

```bash
python -m pip install 'evalscope[lawbench]'
```

若从 EvalScope 源代码检出目录运行，请改用 `python -m pip install -e '.[lawbench]'`。
以下示例评估法律考试子集；它不需要 ChERRANT：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-model', eval_type='openai_api',
    api_url='http://localhost:8000/v1', api_key='EMPTY',
    datasets=['lawbench'],
    dataset_args={'lawbench': {'subset_list': ['1-2']}},
    limit=2,
    generation_config={'temperature': 0, 'max_tokens': 1024},
))
```

### 配置文档校对（任务 2-1）

任务 `2-1` 使用 ChERRANT 比较文本更正并在评估记录上计算 F0.5。它需要额外的 Python 包和官方工具文件；CPU 执行即可满足需求。其他 19 项任务不需要此配置。

安装可选依赖项并下载兼容的工具版本：

```bash
python -m pip install 'evalscope[lawbench-cherrant]'
git clone --filter=blob:none --sparse https://github.com/open-compass/LawBench.git
git -C LawBench sparse-checkout set evaluation/utils
git -C LawBench checkout e30981bb3ff54c41571f222e0b23e92d27375388
```

若从 EvalScope 源代码检出目录运行，请将第一条命令替换为 `python -m pip install -e '.[lawbench-cherrant]'`。
请在包含下载的 `LawBench` 文件夹的目录中运行以下示例。若要评估所有 20 项任务，请移除 `subset_list`。

```python
from pathlib import Path

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-model', eval_type='openai_api',
    api_url='http://localhost:8000/v1', api_key='EMPTY',
    datasets=['lawbench'],
    dataset_args={'lawbench': {
        'subset_list': ['2-1'],
        'extra_params': {
            'cherrant_path': str(Path('LawBench/evaluation/utils').resolve()),
            # 'cherrant_python': '/absolute/path/to/cherrant-env/bin/python',
        },
    }},
    limit=2,
    generation_config={'temperature': 0, 'max_tokens': 1024},
))
```

如果这些依赖项安装在单独的 Python 环境中，请将 `extra_params.cherrant_python` 设置为该环境的绝对 Python 可执行文件路径。请使用该解释器的 `python -m pip` 安装可选依赖项；创建单独的环境是可选的。
下面的通用 Usage 示例选择了所有任务，因此也需要如上所示配置 `cherrant_path`。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `lawbench` |
| **数据集ID** | [evalscope/LawBench](https://modelscope.cn/datasets/evalscope/LawBench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2309.16289) |
| **标签** | `Knowledge`, `QA`, `Reasoning` |
| **指标** | `lawbench_score`, `abstention_rate` |
| **默认 Shots** | 0-shot |
| **评估划分** | `zero_shot` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 10,000 |
| 提示词长度（平均值） | 391.46 字符 |
| 提示词长度（最小/最大） | 32 / 23490 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词均值 | 提示词最小值 | 提示词最大值 |
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
      "id": "d793572c",
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
| `cherrant_timeout` | `float` | `600.0` | 每个校对评分步骤的最大等待时间（秒）。 |

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
