# AIR-Bench-Chat

## 概述

AIR-Bench Chat 是 [AIR-Bench](https://arxiv.org/abs/2402.07729)（音频指令基准，ACL 2024 主会议）的生成部分 —— 这是第一个针对大型音频语言模型（LALMs）的指令跟随基准测试，涵盖了**人类语音、自然声音和音乐**。它包含大约 2k 个开放式音频问答对，涵盖语音、声音、音乐和混合音频场景；响应由 GPT-4 评委根据参考答案进行评分。

## 任务描述

- **任务类型**：开放式音频问答。
- **输入**：一个音频片段加上自由形式的问题。
- **输出**：与参考响应对比评估的文本答案。
- **模态**：音频（人类语音、自然声音、音乐）+ 文本。

## 主要特点

- 约 2k 个跨语音、声音、音乐和混合音频场景的开放式音频问答对；AIR-Bench 的生成部分（ACL 2024）。
- 8 个聊天任务通过官方 `cal_score.py` 聚合为 5 个报告类别：`speech` (`speech_QA`, `speech_dialogue_QA`)，`sound` (`sound_QA`, `sound_generation_QA`)，`music` (`music_QA`, `music_generation_analysis_QA`)，`speech_and_sound` (`speech_and_sound_QA`)，`speech_and_music` (`speech_and_music_QA`)。论文中的混合音频 = mean(speech_and_sound, speech_and_music)。
- 通过将每个样本用参考/预测顺序交换两次后平均来消除位置偏差（可以通过 `extra_params={'do_swap': False}` 禁用来减少评委成本的一半）。
- 托管在 ModelScope ([`evalscope/AIR-Bench-Dataset`](https://modelscope.cn/datasets/evalscope/AIR-Bench-Dataset)) 上，采用音频文件夹 + JSON 布局；完整发布约为 49 GB，因此可以通过 `extra_params={'tasks': [...]}` 限制任务以进行部分运行。

## 评估说明

- 指标：`judge_score` 是模型的平均评委得分；`win_rate` 记录模型严格优于参考答案的频率。
- 评委大模型接收问题、文本音频描述 (`meta_info`)、参考答案 (`answer_gt`) 和模型的响应，并输出两个在 `[1, 10]` 范围内的整数分数。使用支持长上下文的评委，因为对话任务的 `meta_info` 可能超过 4k 个令牌。
- 官方排行榜使用 `gpt-4-0125-preview`。如果该特定快照不可用，请使用可用的 GPT-4 类评委；由于评委模型的变化，绝对分数可能会与已发布的数字有所不同。
- 如果数据集已经在磁盘上，传递 `dataset_args={'air_bench_chat': {'local_path': '/path/to/AIR-Bench-Dataset'}}`；本地根目录应包含 `Chat/`。
- 需要一个支持 `input_audio` 内容部分的 OpenAI 兼容服务器以处理音频输入。SGLang 在 v0.5.19 (sgl-project/sglang#33606) 中添加了此支持；早期版本会以 HTTP 400 拒绝音频请求。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `air_bench_chat` |
| **数据集ID** | [evalscope/AIR-Bench](https://modelscope.cn/datasets/evalscope/AIR-Bench/summary) |
| **论文** | [论文](https://aclanthology.org/2024.acl-long.109/) |
| **标签** | `Audio`, `InstructionFollowing`, `QA` |
| **指标** | `judge_score`, `win_rate` |
| **默认示例** | 0-shot |
| **评估拆分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,200 |
| 提示词长度（平均值） | 83.89 字符 |
| 提示词长度（最小/最大值） | 17 / 423 字符 |

**子集统计数据：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `speech_QA` | 400 | 64.33 | 23 | 148 |
| `speech_dialogue_QA` | 400 | 77.03 | 29 | 206 |
| `sound_QA` | 400 | 73.29 | 17 | 166 |
| `sound_generation_QA` | 100 | 222.52 | 130 | 423 |
| `music_QA` | 400 | 57.54 | 24 | 202 |
| `music_generation_analysis_QA` | 100 | 267.52 | 148 | 395 |
| `speech_and_sound_QA` | 200 | 63.98 | 25 | 127 |
| `speech_and_music_QA` | 200 | 69.37 | 32 | 127 |

**音频统计数据：**

| 指标 | 值 |
|--------|-------|
| 总音频文件数 | 2,200 |
| 每个样本的音频数 | 最小: 1, 最大: 1, 平均: 1 |
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

**提示模板:**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `tasks` | `list` | `None` | 可选的聊天任务名称列表，用于评估（['music_QA', 'music_generation_analysis_QA', 'sound_QA', 'sound_generation_QA', 'speech_QA', 'speech_and_music_QA', 'speech_and_sound_QA', 'speech_dialogue_QA'] 的子集）。默认为所有任务。 |
| `do_swap` | `bool` | `True` | 当为 True（默认值）时，每个样本会被评委评审两次，参考答案和预测顺序交换，然后取平均分。禁用此选项可以减少评委成本，但会有位置偏差。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets air_bench_chat \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
    --limit 10  # Remove this line for formal evaluation
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
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    dataset_args={
        'air_bench_chat': {
            # subset_list: ['speech_QA', 'speech_dialogue_QA', 'sound_QA']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
