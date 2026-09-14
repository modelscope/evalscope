# THCHS-30


## 概述

THCHS-30 是一个带有音素级时间对齐信息的普通话朗读语音语料库。本 EvalScope 基准测试利用该数据集发布的 IPA 音素序列，评估从语音中识别音素的能力。

## 任务描述

- **任务类型**：音素识别（Phoneme Recognition）
- **输入**：一段 16 kHz 的普通话语音录音
- **输出**：带声调符号的 IPA 音素标记序列，以空格分隔
- **领域**：普通话朗读语音

## 主要特性

- 共包含 13,388 条话语，划分为 10,000 条训练样本、893 条验证样本和 2,495 条测试样本
- 提供音素级起止时间戳，包括 `[SIL]` 静音区间
- 保留原始汉字转录文本、说话人标识符、音频时长及划分标签作为样本元数据
- 测试评估使用数据集发布的 IPA 音素表及声调标注

## 评估说明

- 默认配置仅评估 `test` 划分；不使用少样本（few-shot）示例
- 主要指标：语料级音素错误率（Phone Error Rate, PER），计算方式为总音素编辑距离除以参考音素总数；报告中的计数仍以话语数量为准
- 模型输出必须仅为以空格分隔的 IPA 音素标记；时间戳仅作为下游分析的元数据，而非模型目标
- 音频通过 EvalScope 标准音频消息格式以 WAV 数据形式传入

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `thchs30` |
| **数据集ID** | [evalscope/THCHS-30](https://modelscope.cn/datasets/evalscope/THCHS-30/summary) |
| **论文** | [Paper](https://arxiv.org/abs/1512.01882) |
| **标签** | `Audio`, `Chinese`, `SpeechRecognition` |
| **指标** | `per` |
| **默认样本数** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `weighted_mean` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,495 |
| 提示词长度（平均） | 88 字符 |
| 提示词长度（最小/最大） | 88 / 88 字符 |

**音频统计：**

| 指标 | 值 |
|--------|-------|
| 音频文件总数 | 2,495 |
| 每样本音频数 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | wav |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "307a05d8",
      "content": [
        {
          "text": "Transcribe the audio as IPA phone tokens. Output only tokens separated by single spaces."
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~284.9KB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "[SIL] t a˥ ŋ ɻ a˧˥ː n [SIL] tʰ˘ a˥˘ m˘ ə n˘ p u˥˩˘ ɕ j a˥˩˘ ŋ ɤ˥˩ː y˧˥ n a˥˩ jˑ a˥˩˘ ŋ˘ tʰ w˘ ə˥˘ n ɕ˘ j a˥˩ ʂː ɻ̩˧˥ kʰ w ai̯˥˩ˑ [SIL] t a˥ ŋ ts˘ wˑ o˥˩ˑ jː a˥ tsʰ a˥ ŋ ʂː ɻ̩˧˥ˑ [SIL] ɚ˧˥˘ ʂ ɻ̩˥˩ mˑ w o˧˥ ʈʂʰ˘ ə˧˥ ŋ f ə˧˩˧ nː i˧˩˧ xː ou̯˥˩ tsʰ˘ ai̯˧˥˘ fˑ u˧˥ˑ j ʊ˥˩ ŋ [SIL]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "utt_id": "D11_779",
    "text": "当然 他们 不 像 鳄鱼 那样 吞下 石块 当做 压 舱 石 而是 磨成 粉 以后 才 服用",
    "phones": [
      "[SIL]",
      "t",
      "a˥",
      "ŋ",
      "ɻ",
      "a˧˥ː",
      "n",
      "[SIL]",
      "tʰ˘",
      "a˥˘",
      "... [TRUNCATED 66 more items] ..."
    ],
    "phone_starts": [
      0.0,
      1.55,
      1.6,
      1.69,
      1.75,
      1.81,
      1.96,
      2.03,
      2.33,
      2.39,
      "... [TRUNCATED 66 more items] ..."
    ],
    "phone_ends": [
      1.55,
      1.6,
      1.69,
      1.75,
      1.81,
      1.96,
      2.03,
      2.33,
      2.39,
      2.43,
      "... [TRUNCATED 66 more items] ..."
    ],
    "language": "cmn",
    "speaker_id": "D11",
    "duration": 9.114,
    "split": "test"
  }
}
```

*注：部分内容因显示需要已被截断。*

## 提示模板

**提示模板：**
```text
Transcribe the audio as IPA phone tokens. Output only tokens separated by single spaces.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets thchs30 \
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
    datasets=['thchs30'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
