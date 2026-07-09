# TVBench


## 概述

TVBench 是一个用于评估多模态模型是否能够对动态视觉事件（而非孤立帧）进行推理的时序视频理解基准测试。它涵盖了广泛的视频推理能力，包括动作识别、动作计数、时序定位、动作顺序理解、第一人称视角动作排序、物体计数、物体重排、运动方向识别、场景转换推理以及异常动作检测。

EvalScope 原生适配器从数据集仓库中读取官方提供的按任务划分的 JSON 注释，并按需解析对应的视频压缩包。这种方式使得默认的冒烟测试路径保持轻量，同时仍可通过 `subset_list` 支持完整的基准测试。

## 任务描述

- **任务类型**：视频多项选择题问答（MCQ）
- **输入**：一段视频片段或带时间范围的视频片段、一个自然语言问题以及 2-4 个候选答案
- **输出**：从提供的候选答案中选择一个字母选项作为答案
- **默认子集**：`action_count`，因其在公开数据集仓库中以标准 MP4 压缩包形式提供
- **支持的子集**：`action_antonym`、`action_count`、`action_localization`、`action_sequence`、`egocentric_sequence`、`moving_direction`、`object_count`、`object_shuffle`、`scene_transition` 和 `unexpected_action`

## 评估说明

- 默认评估使用 0-shot 思维链（Chain-of-Thought）多项选择提示模板 `MultipleChoiceTemplate.SINGLE_ANSWER_COT`。
- 主要指标：准确率（`acc`）。数据集中的答案以候选文本形式存储，在评分前会转换为对应的选项字母。
- 部分子集提供 `start`/`end` 字段。适配器会将这些值传递给 `ContentVideo`，并在提示中添加简洁的片段说明。
- 视频文件按需从子集特定的压缩包中懒加载下载。`egocentric_sequence` 使用位于 `video/egocentric_sequence/<prefix>.zip` 下的分段压缩包。
- `action_antonym` 的注释引用 AVI 文件。如果仓库媒体压缩包不可用，请通过配置 `extra_params.video_dir` 指向包含 AVI 文件的本地目录。
- 适配器支持两种本地视频布局：扁平结构（`video_dir/<video>`）或按子集分组结构（`video_dir/<subset>/<video>`）。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tvbench` |
| **数据集ID** | [evalscope/TVBench](https://modelscope.cn/datasets/evalscope/TVBench/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数（Shots）** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 536 |
| 提示词长度（平均） | 326.44 字符 |
| 提示词长度（最小/最大） | 290 / 358 字符 |

**视频统计信息：**

| 指标 | 值 |
|--------|-------|
| 总视频数 | 536 |
| 每样本视频数 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | mp4 |


## 样例示例

**子集**: `action_count`

```json
{
  "input": [
    {
      "id": "4b521c81",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nThe person makes sets of repeated actions. How many times did the person repeat the action in the last set?\n\nA) 2\nB) 3\nC) 5\nD) 4"
        },
        {
          "video": "~/.cache/modelscope/hub/datasets/tvbench/videos/action_count/video_8686.mp4",
          "format": "mp4"
        }
      ]
    }
  ],
  "choices": [
    "2",
    "3",
    "5",
    "4"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "video": "video_8686.mp4",
    "answer": "4",
    "subset": "action_count",
    "start": null,
    "end": null,
    "fps": null,
    "accurate_start": null,
    "accurate_end": null,
    "video_length": null,
    "question_id": null,
    "dataset_id": "evalscope/TVBench",
    "dataset_hub": "modelscope"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `video_dir` | `str` | `` | 可选的本地目录，包含 TVBench 视频文件。该目录可以是扁平结构，也可以按子集组织。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tvbench \
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
    datasets=['tvbench'],
    dataset_args={
        'tvbench': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
