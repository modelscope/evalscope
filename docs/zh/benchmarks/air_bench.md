# AIR-Bench

## 概述

[AIR-Bench](https://arxiv.org/abs/2402.07729)（Audio InstRuction Benchmark，ACL 2024 主会议）是首个面向大型音频-语言模型（LALM）的指令型评测基准，覆盖**人类语音、自然声音、音乐**三大类音频信号，并能直接对开放式回答进行评分。

AIR-Bench 拆分为两条评测轨道，evalscope 分别注册为：

| Benchmark 名称 | 题型 | 题量 | 评分方式 |
|---|---|---|---|
| `air_bench_foundation` | 单选题（A/B/C/D） | ~25k | 规则匹配，accuracy |
| `air_bench_chat` | 开放式音频问答 | ~2k | GPT-4 评分（1–10 分），位置互换后取均值 |

## 数据获取

数据托管在 Hugging Face：[`qyang1021/AIR-Bench-Dataset`](https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset)，使用 *audiofolder + JSON 元数据* 的布局。evalscope 在首次运行时通过 `huggingface_hub.snapshot_download` 自动下载；完整数据约 49 GB，建议先用 `extra_params` 限定要评估的子集再下载。

如已离线下载，可通过 `dataset_args={'<benchmark>': {'local_path': '/path/to/AIR-Bench-Dataset'}}` 指向本地根目录（包含 `Foundation/` 和/或 `Chat/` 子目录）。

Foundation 中部分样本是 FLAC。对于 OpenAI-compatible 音频输入，evalscope 会将其转换为缓存 WAV 文件；这需要安装 `soundfile`（`pip install "evalscope[air_bench]"`）或在系统 `PATH` 中提供可用的 `ffmpeg`。

## Foundation 轨道（`air_bench_foundation`）

19 个任务、25 个源数据子目录，按论文聚合为三类：

- **Speech**（11 个子目录 / 9 个任务）：speech grounding、language ID、gender、emotion（IEMOCAP+MELD）、age、speech entity、intent、speaker counting、synthesized voice。
- **Sound**（6 个子目录 / 4 个任务）：audio grounding、vocal sound、acoustic scene（CochlScene+TUT2017）、sound QA（avqa+clothoaqa）。
- **Music**（8 个子目录 / 6 个任务）：instruments、genre、MIDI pitch、MIDI velocity、music QA、music emotion。

### 属性

| 属性 | 值 |
|---|---|
| **基准测试名称** | `air_bench_foundation` |
| **数据集 ID** | [qyang1021/AIR-Bench-Dataset](https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset) |
| **论文** | [AIR-Bench (Yang et al., ACL 2024)](https://aclanthology.org/2024.acl-long.109.pdf) |
| **标签** | `Audio`, `MCQ`, `Knowledge` |
| **指标** | `acc` |
| **评估划分** | `test` |
| **可选参数** | `subsets`：限定要评测的源数据子目录列表 |

### Prompt 模板（与官方 `Inference_Foundation.py` 一致）

```text
Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D.
{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
```

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_AUDIO_LLM \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets air_bench_foundation \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python（仅评估 music 类目）

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_AUDIO_LLM',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['air_bench_foundation'],
    dataset_args={
        'air_bench_foundation': {
            # 'local_path': '/path/to/AIR-Bench-Dataset',  # 可选
            'extra_params': {
                'subsets': [
                    'Music_AQA_music_avqa',
                    'Music_Genre_Recognition_MTJ-Jamendo',
                    'Music_Genre_Recognition_fma',
                    'Music_Instruments_Classfication_MTJ-Jamendo',
                    'Music_Instruments_Classfication_nsynth',
                    'Music_Midi_Pitch_Analysis_nsynth',
                    'Music_Midi_Velocity_Analysis_nsynth',
                    'Music_Mood_Recognition_MTJ-Jamendo',
                ]
            }
        }
    },
)

run_task(task_cfg=task_cfg)
```

## Chat 轨道（`air_bench_chat`）

8 个任务，按官方 `cal_score.py` 聚合为五个类目：`speech`、`sound`、`music`、`speech_and_sound`、`speech_and_music`。论文 leaderboard 中的 **Mixed-audio = mean(speech_and_sound, speech_and_music)**。

### 属性

| 属性 | 值 |
|---|---|
| **基准测试名称** | `air_bench_chat` |
| **数据集 ID** | [qyang1021/AIR-Bench-Dataset](https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset) |
| **论文** | [AIR-Bench (Yang et al., ACL 2024)](https://aclanthology.org/2024.acl-long.109.pdf) |
| **标签** | `Audio`, `QA`, `InstructionFollowing` |
| **指标** | `gpt_score`（平均 1–10 分判分）、`win_rate`（严格优于参考的比例） |
| **评估划分** | `test` |
| **可选参数** | `tasks`：限定要评测的任务名；`do_swap`：是否启用位置互换（默认 `True`） |

### 评分协议

判分模板沿用官方协议，由判分模型对 *参考答案*（Assistant 1）和 *模型答案*（Assistant 2）同时打 1–10 分；为消除位置偏置，每个样本判分两次（默认开启 `do_swap`，第二次互换位置），取均值。

```{warning}
官方 leaderboard 使用 `gpt-4-0125-preview` 作为判分模型。如该 snapshot 不可用，请配置可用的 GPT-4 级别判分模型；由于判分模型变化，绝对分数可能与公开榜单存在偏差。
```

### 使用 CLI（含判分模型配置）

```bash
evalscope eval \
    --model YOUR_AUDIO_LLM \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets air_bench_chat \
    --judge-strategy llm \
    --judge-model-args '{"model_id": "YOUR_GPT4_JUDGE", "api_key": "OPENAI_KEY"}' \
    --limit 10
```

### 使用 Python（关闭位置互换以加速）

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_AUDIO_LLM',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['air_bench_chat'],
    judge_strategy='llm',
    judge_model_args={
        'model_id': 'YOUR_GPT4_JUDGE',
        'api_key': 'OPENAI_KEY',
    },
    dataset_args={
        'air_bench_chat': {
            # 'local_path': '/path/to/AIR-Bench-Dataset',  # 可选
            'extra_params': {
                'tasks': ['speech_QA', 'speech_dialogue_QA'],
                'do_swap': False,
            }
        }
    },
)

run_task(task_cfg=task_cfg)
```
