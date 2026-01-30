# CCBench


## 概述

CCBench（Chinese Culture Bench）是 MMBench 的一个扩展，专门用于评估多模态模型对中华传统文化的理解能力。它通过视觉问答的形式，涵盖中华文化遗产的多个方面。

## 任务描述

- **任务类型**：视觉多项选择题问答（中华文化）
- **输入**：包含中华文化相关问题的图像
- **输出**：单个正确答案字母（A、B、C 或 D）
- **语言**：主要为中文内容

## 主要特点

- 聚焦中华传统文化相关问题
- 类别包括：书法、绘画、文物、饮食与服饰
- 历史人物、风景与建筑、草图推理、传统表演
- 测试结合视觉理解与文化知识的能力
- 基于 MMBench 评估框架的扩展

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用思维链（Chain-of-Thought, CoT）提示
- 在测试集（test split）上进行评估
- 使用简单准确率（accuracy）作为评分指标
- 要求模型同时具备视觉感知能力和文化知识

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `cc_bench` |
| **数据集ID** | [lmms-lab/MMBench](https://modelscope.cn/datasets/lmms-lab/MMBench/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,040 |
| 提示词长度（平均） | 270.1 字符 |
| 提示词长度（最小/最大） | 254 / 394 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 2,040 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 119x118 - 512x512 |
| 格式 | jpeg |


## 样例示例

**子集**: `cc`

```json
{
  "input": [
    {
      "id": "2797f551",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\n图中所示建筑名称为？\n\nA) 天坛\nB) 故宫\nC) 黄鹤楼\nD) 少林寺"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~22.7KB]"
        }
      ]
    }
  ],
  "choices": [
    "天坛",
    "故宫",
    "黄鹤楼",
    "少林寺"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "category": "scenery_building",
    "source": "https://zh.wikipedia.org/wiki/%E5%A4%A9%E5%9D%9B"
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

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cc_bench \
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
    datasets=['cc_bench'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```