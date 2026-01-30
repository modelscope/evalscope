# TORGO

## 概述

TORGO 是一个专门用于评估自动语音识别（ASR）系统在运动性言语障碍患者上的表现的数据库。它包含来自脑瘫（CP）或肌萎缩侧索硬化症（ALS）患者的对齐声学与发音数据。

## 任务描述

- **任务类型**：运动性构音障碍语音识别  
- **输入**：言语障碍者的音频录音  
- **输出**：转录文本  
- **重点**：无障碍与包容性 ASR 评估  

## 主要特性

- 针对运动性构音障碍语音的专用数据集  
- 包含脑瘫（CP）或 ALS 患者的数据  
- 基于可懂度划分的子集（轻度、中度、重度）  
- 3D 发音特征对齐  
- 对无障碍研究具有重要意义  

## 评估说明

- 默认配置使用 **test** 数据划分  
- 按可懂度划分的子集：**mild**（轻度）、**moderate**（中度）、**severe**（重度）  
- 评估指标：**CER**（字符错误率）、**WER**（词错误率）、**SemScore**  
- CER/WER 指标需安装 `jiwer` 包  
- SemScore 指标需安装 `jellyfish` 包  
- 支持批量评分以提升效率  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `torgo` |
| **数据集ID** | [extraordinarylab/torgo](https://modelscope.cn/datasets/extraordinarylab/torgo/summary) |
| **论文** | N/A |
| **标签** | `Audio`, `SpeechRecognition` |
| **指标** | `cer`, `wer`, `sem_score` |
| **默认提示方式** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 5,553 |
| 提示词长度（平均） | 67 字符 |
| 提示词长度（最小/最大） | 67 / 67 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `mild` | 1,479 | 67 | 67 | 67 |
| `moderate` | 1,666 | 67 | 67 | 67 |
| `severe` | 2,408 | 67 | 67 | 67 |

**音频统计：**

| 指标 | 值 |
|--------|-------|
| 音频文件总数 | 5,553 |
| 每样本音频数量 | 最小: 1, 最大: 1, 平均: 1 |
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

**提示模板：**
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
    datasets=['torgo'],
    dataset_args={
        'torgo': {
            # subset_list: ['mild', 'moderate', 'severe']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```