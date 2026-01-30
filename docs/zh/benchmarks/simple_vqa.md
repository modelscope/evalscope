# SimpleVQA

## 概述

SimpleVQA 是首个全面的多模态基准测试，用于评估多模态大语言模型（MLLMs）回答自然语言简短问题时的事实性能力。该基准包含高质量、具有挑战性的问题，并配有静态且不受时间影响的标准答案。

## 任务描述

- **任务类型**：事实性视觉问答（Factual Visual Question Answering）
- **输入**：图像 + 事实性问题
- **输出**：简短的事实性答案
- **领域**：事实性、视觉推理、知识回忆

## 主要特点

- 覆盖多种任务和场景
- 高质量、具有挑战性的问题
- 静态且不受时间影响的标准答案（无时间依赖性）
- 评估方法直接明了
- 测试模型真实的事实性知识，而非仅依赖模式匹配

## 评估说明

- 默认使用 **test** 数据划分进行评估
- 主要指标：基于大语言模型（LLM）评判器的 **准确率（Accuracy）**
- 三级评分标准：CORRECT（正确）、INCORRECT（错误）、NOT_ATTEMPTED（未作答）
- LLM 评判器采用详细的评分规则进行语义匹配
- 包含丰富的元数据，如语言、来源和原子事实

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `simple_vqa` |
| **数据集ID** | [m-a-p/SimpleVQA](https://modelscope.cn/datasets/m-a-p/SimpleVQA/summary) |
| **论文** | N/A |
| **标签** | `MultiModal`, `QA`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,025 |
| 提示词长度（平均） | 56.22 字符 |
| 提示词长度（最小/最大） | 27 / 1015 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 2,025 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 106x56 - 5119x3413 |
| 格式 | jpeg, png |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "4340fc24",
      "content": [
        {
          "text": "Answer the question:\n\n图中所示穴位所属的经脉是什么？"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~26.5KB]"
        }
      ]
    }
  ],
  "target": "足阳明胃经",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "data_id": 0,
    "image_description": "",
    "language": "CN",
    "original_category": "中华文化_中医",
    "source": "https://baike.baidu.com/item/%E4%BC%8F%E5%85%94%E7%A9%B4/3503684#:~:text\\u003d%E4%BA%BA%E4%BD%93%E7%A9%B4%E4%BD%8D%E5%90%8D%E4%BC%8F%E5%85%94%E7%A9%B4F%C3%BA%20t%C3%B9%EF%BC%88ST32%EF%BC%89%E5%B1%9E%E8%B6%B3%E9%98%B3%E6%98%8E%E8%83%83%E7%BB%8 ... [TRUNCATED] ... 4%BE%A7%E7%AB%AF%E7%9A%84%E8%BF%9E%E7%BA%BF%E4%B8%8A%EF%BC%8C%E9%AB%8C%E9%AA%A8%E4%B8%8A%E7%BC%98%E4%B8%8A6%E5%AF%B8%E3%80%82%E4%BC%8F%E5%85%94%E5%88%AB%E5%90%8D%E5%A4%96%E4%B8%98%E3%80%81%E5%A4%96%E5%8B%BE%EF%BC%8C%E4%BD%8D%E4%BA%8E%E5%A4%A7",
    "atomic_question": "图中所示穴位的名称是什么？",
    "atomic_fact": "伏兔"
  }
}
```

*注：部分内容因显示需要已被截断。*

## 提示模板

**提示模板：**
```text
Answer the question:

{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets simple_vqa \
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
    datasets=['simple_vqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```