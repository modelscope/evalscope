# AIR-Bench-Foundation


## 概述

AIR-Bench Foundation 是 [AIR-Bench](https://arxiv.org/abs/2402.07729)（Audio InstRuction Benchmark，ACL 2024 主会）的判别任务部分——这是首个面向大型音频-语言模型（LALMs）的指令遵循基准测试，涵盖**人类语音、自然声音和音乐**。Foundation 轨道包含约 2.5 万道单选题，横跨三大音频类别中的 19 项逻辑任务。

## 任务描述

- **任务类型**：基于音频片段的单选问答。
- **输入**：一段音频 + 一道包含最多四个候选答案（A/B/C/D）的问题。
- **输出**：从给定选项中选择一个字母。

## 类别（19 项任务 / 25 个源数据集子集）

- **语音**（11 个子集 / 9 项任务）：语音定位、语言识别、性别识别、情感识别（IEMOCAP+MELD）、年龄预测、语音实体识别、意图分类、说话人数量统计、合成语音检测。
- **声音**（6 个子集 / 4 项任务）：音频定位、人声分类、声学场景分类（CochlScene+TUT2017）、声音问答（avqa+clothoaqa）。
- **音乐**（8 个子集 / 6 项任务）：乐器识别、流派识别、MIDI 音高分析、MIDI 力度分析、音乐问答、音乐情感识别。

## 提示模板（与官方 `Inference_Foundation.py` 一致）

```text
Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D.
{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
```

## 数据集获取

- 该数据集托管于 ModelScope：[`evalscope/AIR-Bench-Dataset`](https://modelscope.cn/datasets/evalscope/AIR-Bench-Dataset)。采用 *audiofolder + JSON metadata* 的布局。evalscope 在首次运行时通过 `modelscope.dataset_snapshot_download` 按需下载；完整数据集约 49 GB，建议通过 `extra_params` 参数限制下载的子集。
- 如果数据集已存在于本地磁盘，请传入 `dataset_args={'air_bench_foundation': {'local_path': '/path/to/AIR-Bench-Dataset'}}`；本地根目录应包含 `Foundation/` 文件夹。
- 部分 Foundation 样本为 FLAC 格式。为兼容 OpenAI 风格的音频输入，evalscope 会将其转换为缓存的 WAV 文件，这需要安装 `soundfile`（`pip install "evalscope[air_bench]"`）或系统中存在可用的 `ffmpeg` 二进制文件。

## 评估说明

- **指标**：**准确率**（按源数据集子集计算，并按类别聚合）。
- 默认提示模板遵循官方 `Inference_Foundation.py` 的格式，因此可复现现有的 AIR-Bench 排行榜结果。
- 设置 `extra_params={'subsets': [...]}` 可限制评估范围至 25 个源目录中的部分子集——适用于仅下载了部分数据的情况。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `air_bench_foundation` |
| **数据集 ID** | [evalscope/AIR-Bench](https://modelscope.cn/datasets/evalscope/AIR-Bench/summary) |
| **论文** | [Paper](https://aclanthology.org/2024.acl-long.109/) |
| **标签** | `Audio`, `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 21,426 |
| 提示词长度（平均） | 236.68 字符 |
| 提示词长度（最小/最大） | 179 / 321 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Speaker_Age_Prediction_common_voice_13.0_en` | 1,000 | 291.98 | 278 | 305 |
| `Speaker_Emotion_Recontion_iemocap` | 1,000 | 227.13 | 218 | 238 |
| `Speaker_Emotion_Recontion_meld` | 1,000 | 233.21 | 218 | 250 |
| `Speaker_Gender_Recognition_common_voice_13_en` | 780 | 226.19 | 213 | 241 |
| `Speaker_Gender_Recognition_meld` | 1,000 | 226.42 | 213 | 241 |
| `Speaker_Intent_Classification_slurp` | 662 | 268.52 | 232 | 295 |
| `Speaker_Number_Verification_voxceleb1` | 314 | 208.24 | 194 | 221 |
| `Speech_Entity_Reconition_slurp` | 1,000 | 253.14 | 226 | 316 |
| `Speech_Grounding_librispeech` | 981 | 253.92 | 230 | 282 |
| `Spoken_Language_Identification_covost2` | 495 | 207.12 | 191 | 232 |
| `Synthesized_Voice_Detection_fake_or_real` | 1,000 | 224.79 | 203 | 242 |
| `Acoustic_Scene_Classification_CochlScene` | 1,000 | 240.97 | 213 | 278 |
| `Acoustic_Scene_Classification_TUT2017` | 1,000 | 241.72 | 207 | 284 |
| `Audio_Grounding_AudioGrounding` | 896 | 273.74 | 249 | 321 |
| `Sound_AQA_avqa` | 1,000 | 227.21 | 193 | 298 |
| `Sound_AQA_clothoaqa` | 1,000 | 199.89 | 179 | 262 |
| `vocal_sound_classification_VocalSound` | 985 | 232.62 | 210 | 253 |
| `Music_AQA_music_avqa` | 814 | 208.7 | 185 | 238 |
| `Music_Genre_Recognition_MTJ-Jamendo` | 1,000 | 223.84 | 200 | 248 |
| `Music_Genre_Recognition_fma` | 1,000 | 224.59 | 201 | 250 |
| `Music_Instruments_Classfication_MTJ-Jamendo` | 1,000 | 236.52 | 218 | 262 |
| `Music_Instruments_Classfication_nsynth` | 996 | 228.12 | 216 | 247 |
| `Music_Midi_Pitch_Analysis_nsynth` | 493 | 253.88 | 243 | 264 |
| `Music_Midi_Velocity_Analysis_nsynth` | 484 | 270.6 | 259 | 279 |
| `Music_Mood_Recognition_MTJ-Jamendo` | 526 | 229.67 | 210 | 248 |

**音频统计信息：**

| 指标 | 值 |
|--------|-------|
| 音频文件总数 | 21,426 |
| 每样本音频数量 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | mp3, wav |


## 样例示例

**子集**: `Speaker_Age_Prediction_common_voice_13.0_en`

```json
{
  "input": [
    {
      "id": "544443c0",
      "content": [
        {
          "audio": "[BASE64_AUDIO: mp3, ~25.9KB]",
          "format": "mp3"
        },
        {
          "text": "Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D.\nWhich age range do you believe best matches the speaker's voice?\nA. teens to twenties\nB. thirties to fourties\nC. fifties to sixties\nD. seventies to eighties"
        }
      ]
    }
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "Speaker_Age_Prediction_common_voice_13.0_en",
  "metadata": {
    "uniq_id": 5973,
    "task_name": "Speaker_Age_Prediction",
    "dataset_name": "common_voice_13.0_en",
    "category": "speech",
    "answer_gt_text": "thirties to fourties",
    "choices": {
      "A": "teens to twenties",
      "B": "thirties to fourties",
      "C": "fifties to sixties",
      "D": "seventies to eighties"
    }
  }
}
```

## 提示模板

**提示模板：**
```text
Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D.
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `subsets` | `list` | `None` | 可选的 Foundation 源数据集目录列表，用于指定评估范围。默认为全部 25 个目录。当本地仅下载了部分子集时非常有用。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets air_bench_foundation \
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
    datasets=['air_bench_foundation'],
    dataset_args={
        'air_bench_foundation': {
            # subset_list: ['Speaker_Age_Prediction_common_voice_13.0_en', 'Speaker_Emotion_Recontion_iemocap', 'Speaker_Emotion_Recontion_meld']  # 可选，评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```