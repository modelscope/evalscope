# AIR-Bench-Chat


## 概述

AIR-Bench Chat 是 [AIR-Bench](https://arxiv.org/abs/2402.07729)（Audio InstRuction Benchmark，ACL 2024 主会）的生成部分——这是首个面向大型音频-语言模型（LALMs）的指令遵循基准测试，涵盖**人类语音、自然声音和音乐**。该数据集包含约 2,000 个开放式音频问答对，覆盖语音、声音、音乐及混合音频场景；模型的回答由 GPT-4 作为裁判，对照参考答案进行评分。

## 任务描述

- **任务类型**：开放式音频问答。
- **输入**：一段音频片段加一个自由形式的问题。
- **输出**：一段文本答案，将根据参考答案进行评估。

## 类别（8 项任务 → 5 个报告类别）

官方 `cal_score.py` 脚本将 8 项 Chat 任务聚合为以下五个类别：

- `speech`（语音）: `speech_QA`, `speech_dialogue_QA`
- `sound`（声音）: `sound_QA`, `sound_generation_QA`
- `music`（音乐）: `music_QA`, `music_generation_analysis_QA`
- `speech_and_sound`（语音与声音）: `speech_and_sound_QA`
- `speech_and_music`（语音与音乐）: `speech_and_music_QA`

论文中的 **Mixed-audio（混合音频）= mean(speech_and_sound, speech_and_music)**。

## 数据集获取

- 该数据集托管在 ModelScope 上：[`evalscope/AIR-Bench-Dataset`](https://modelscope.cn/datasets/evalscope/AIR-Bench-Dataset)。采用 *audiofolder + JSON 元数据* 的布局。evalscope 在首次运行时通过 `modelscope.dataset_snapshot_download` 按需下载；完整数据集约 49 GB，建议通过 `extra_params` 参数限制仅拉取所需任务。
- 如果数据集已存在于本地磁盘，请传入 `dataset_args={'air_bench_chat': {'local_path': '/path/to/AIR-Bench-Dataset'}}`；本地根目录应包含 `Chat/` 文件夹。

## 评估协议

- 裁判大语言模型（默认为 GPT-4）接收问题、音频的文本描述（来自数据集的 `meta_info`）、参考答案（`answer_gt`）以及模型的回答，并输出一行包含两个 `[1, 10]` 范围内的整数分数。
- 为消除位置偏差，每个样本会被评判两次（交换参考答案与模型预测的顺序），然后取平均分。此做法与官方仓库中的 `cal_score.py` 一致；可通过设置 `extra_params={'do_swap': False}` 禁用该行为，以节省一半的裁判成本。
- 报告指标 `gpt_score` 为模型的平均裁判分数；`win_rate` 记录模型严格优于参考答案的频率。

```{warning}
官方排行榜使用 `gpt-4-0125-preview` 作为裁判模型。如果该特定快照不可用，请使用其他可用的 GPT-4 级别裁判模型；由于裁判模型变更，绝对分数可能与已发表结果存在偏差。
```

## 实现说明

- 通过 `--judge-model-args` 指定裁判模型；请确保所选模型支持长上下文（对话类任务的 `meta_info` 可能超过 4k tokens）。
- 设置 `extra_params={'tasks': [...]}` 可仅评估指定的 Chat 任务名称——适用于部分运行。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `air_bench_chat` |
| **数据集 ID** | [evalscope/AIR-Bench](https://modelscope.cn/datasets/evalscope/AIR-Bench/summary) |
| **论文** | [Paper](https://aclanthology.org/2024.acl-long.109/) |
| **标签** | `Audio`, `InstructionFollowing`, `QA` |
| **指标** | `gpt_score`, `win_rate` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,200 |
| 提示词长度（平均） | 83.89 字符 |
| 提示词长度（最小/最大） | 17 / 423 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `speech_QA` | 400 | 64.33 | 23 | 148 |
| `speech_dialogue_QA` | 400 | 77.03 | 29 | 206 |
| `sound_QA` | 400 | 73.29 | 17 | 166 |
| `sound_generation_QA` | 100 | 222.52 | 130 | 423 |
| `music_QA` | 400 | 57.54 | 24 | 202 |
| `music_generation_analysis_QA` | 100 | 267.52 | 148 | 395 |
| `speech_and_sound_QA` | 200 | 63.98 | 25 | 127 |
| `speech_and_music_QA` | 200 | 69.37 | 32 | 127 |

**音频统计信息：**

| 指标 | 值 |
|--------|-------|
| 音频文件总数 | 2,200 |
| 每样本音频数量 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | mp3, wav |


## 样例示例

**子集**: `speech_QA`

```json
{
  "input": [
    {
      "id": "5781ee73",
      "content": [
        {
          "audio": "/root/.cache/modelscope/hub/datasets/evalscope/AIR-Bench-Dataset/Chat/speech_QA_iemocap/Ses01F_script01_1_M025.wav",
          "format": "wav"
        },
        {
          "text": "Who is the speaker addressing at the end of the speech?"
        }
      ]
    }
  ],
  "target": "The speaker is addressing Mom at the end of the speech.",
  "id": 0,
  "group_id": 0,
  "subset_key": "speech_QA",
  "metadata": {
    "uniq_id": 400,
    "task_name": "speech_QA",
    "dataset_name": "iemocap",
    "category": "speech",
    "meta_info": "{'emotion': 'neutral', 'gender': 'male', 'transcription': \"And then we'll thrash it out with father. Okay Mom? Don't avoid me.\"}",
    "question": "Who is the speaker addressing at the end of the speech?"
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `tasks` | `list` | `None` | 可选的 Chat 任务名称列表（子集包括 ['music_QA', 'music_generation_analysis_QA', 'sound_QA', 'sound_generation_QA', 'speech_QA', 'speech_and_music_QA', 'speech_and_sound_QA', 'speech_dialogue_QA']）。默认评估所有任务。 |
| `do_swap` | `bool` | `True` | 若为 True（默认），每个样本会评判两次（交换参考答案与模型预测的顺序），然后取平均分。禁用可节省一半裁判成本，但会引入位置偏差。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets air_bench_chat \
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
    datasets=['air_bench_chat'],
    dataset_args={
        'air_bench_chat': {
            # subset_list: ['speech_QA', 'speech_dialogue_QA', 'sound_QA']  # 可选，仅评估指定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```