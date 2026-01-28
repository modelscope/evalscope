# MusicTrivia

## 概述

MusicTrivia 是一个精心策划的多项选择题基准测试，用于评估 AI 模型在音乐知识方面的能力。它涵盖古典音乐和现代音乐主题，包括作曲家、音乐时期、乐器和流行艺术家等内容。

## 任务描述

- **任务类型**：多项选择题问答（Multiple-Choice Question Answering）
- **输入**：与音乐相关的常识问题及多个选项
- **输出**：选择正确答案
- **领域**：古典音乐、现代音乐、音乐史

## 主要特点

- 全面覆盖各类音乐领域
- 包含关于作曲家、音乐时期和艺术家的问题
- 考察事实记忆与领域知识
- 精心筛选以确保质量和准确性
- 各主题难度均衡

## 评估说明

- 默认配置使用 **0-shot** 评估方式
- 主要指标：**准确率（Accuracy）**
- 在 **test** 数据集划分上进行评估
- 使用标准的单答案多项选择题模板

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `music_trivia` |
| **数据集 ID** | [extraordinarylab/music-trivia](https://modelscope.cn/datasets/extraordinarylab/music-trivia/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估数据划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 512 |
| 提示词长度（平均） | 412.5 字符 |
| 提示词长度（最小/最大） | 256 / 1005 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "d16392be",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nBeethoven's third period was distinct from his second on account of all the following factors EXCEPT\n\nA) The influence of Italian opera on his compositions\nB) The occupation of Vienna by French troops\nC) The introduction of new instruments in the orchestra\nD) The emergence of Romanticism in music"
    }
  ],
  "choices": [
    "The influence of Italian opera on his compositions",
    "The occupation of Vienna by French troops",
    "The introduction of new instruments in the orchestra",
    "The emergence of Romanticism in music"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets music_trivia \
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
    datasets=['music_trivia'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```