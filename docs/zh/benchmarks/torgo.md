# TORGO

## 概述

TORGO 是一个专门针对运动言语障碍者的言语识别数据库，旨在评估自动语音识别（ASR）系统在患有运动言语障碍的说话者上的表现。它包含了来自脑瘫（CP）或肌萎缩侧索硬化症（ALS）患者的对齐声学和发音数据。

## 任务描述

- **任务类型**: 言语障碍语音识别
- **输入**: 来自言语障碍患者的音频录音
- **输出**: 转录文本
- **重点**: 可访问性和包容性 ASR 评估

## 主要特点

- 专门用于言语障碍语音的数据集
- 患有脑瘫（CP）或 ALS 的说话者
- 基于可理解性的子集（轻度、中度、重度）
- 3D 发音特征对齐
- 对可访问性研究非常重要

## 评估说明

- 默认配置使用 **test** 分割
- 按可理解性划分的子集：**mild**、**moderate**、**severe**
- 指标：**CER**（字符错误率）、**WER**（词错误率）、**SemScore**
- 需要 `jiwer` 包来计算 CER/WER 指标
- 需要 `jellyfish` 包来计算 SemScore 指标
- 支持批量评分以提高效率
- 需要一个支持 `input_audio` 内容部分的 OpenAI 兼容服务器进行音频输入

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `torgo` |
| **数据集ID** | [extraordinarylab/torgo](https://modelscope.cn/datasets/extraordinarylab/torgo/summary) |
| **论文** | N/A |
| **标签** | `Audio`, `SpeechRecognition` |
| **指标** | `cer`, `wer`, `sem_score` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 5,553 |
| 提示词长度（平均） | 67 字符 |
| 提示词长度（最小/最大） | 67 / 67 字符 |

**各子集统计:**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `mild` | 1,479 | 67 | 67 | 67 |
| `moderate` | 1,666 | 67 | 67 | 67 |
| `severe` | 2,408 | 67 | 67 | 67 |

**音频统计:**

| 指标 | 值 |
|--------|-------|
| 总音频文件数 | 5,553 |
| 每个样本的音频数量 | 最小: 1, 最大: 1, 平均: 1 |
| 格式 | wav |

## 样例示例

**子集**: `mild`

```json
{
  "input": [
    {
      "id": "1220f252",
      "content": [
        {
          "text": "Please recognize the speech and only output the recognized content:"
        },
        {
          "audio": "[BASE64_AUDIO: wav, ~89.1KB]",
          "format": "wav"
        }
      ]
    }
  ],
  "target": "FEE",
  "id": 0,
  "group_id": 0,
  "subset_key": "mild",
  "metadata": {
    "transcript": "FEE",
    "intelligibility": "mild",
    "duration": 2.8499999046325684
  }
}
```

## 提示模板

**提示模板:**
```text
Please recognize the speech and only output the recognized content:
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets torgo \
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
    datasets=['torgo'],
    dataset_args={
        'torgo': {
            # subset_list: ['mild', 'moderate', 'severe']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
