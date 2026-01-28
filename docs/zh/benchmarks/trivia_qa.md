# TriviaQA


## 概述

TriviaQA 是一个大规模阅读理解数据集，包含超过 65 万个问题-答案-证据三元组。问题收集自问答爱好者网站，并与维基百科文章配对作为证据文档。

## 任务描述

- **任务类型**：阅读理解 / 问答
- **输入**：带有维基百科上下文段落的问题
- **输出**：从上下文中抽取或生成的答案
- **领域**：通用知识类问答

## 主要特点

- 超过 65 万个问题-答案-证据三元组
- 问题由问答爱好者编写（天然具有挑战性）
- 支持多个有效答案别名，便于灵活评估
- 维基百科文章提供证据段落
- 同时考察阅读理解与知识检索能力

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用维基百科阅读理解子集（rc.wikipedia）
- 答案格式应为："ANSWER: [ANSWER]"
- 支持基于包含关系的答案匹配
- 在验证集（validation split）上进行评估


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `trivia_qa` |
| **数据集ID** | [evalscope/trivia_qa](https://modelscope.cn/datasets/evalscope/trivia_qa/summary) |
| **论文** | N/A |
| **标签** | `QA`, `ReadingComprehension` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 7,993 |
| 提示词长度（平均） | 54126.56 字符 |
| 提示词长度（最小/最大） | 339 / 691325 字符 |

## 样例示例

**子集**: `rc.wikipedia`

```json
{
  "input": [
    {
      "id": "545e4eda",
      "content": "Read the content and answer the following question.\n\nContent: ['Andrew Lloyd Webber, Baron Lloyd-Webber   (born 22 March 1948) is an English composer and impresario of musical theatre. \\n\\nSeveral of his musicals have run for more than a deca ... [TRUNCATED] ... ening titles.']\n\nQuestion: Which Lloyd Webber musical premiered in the US on 10th December 1993?\n\nKeep your The last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the problem.\n"
    }
  ],
  "target": [
    "Sunset Blvd",
    "West Sunset Boulevard",
    "Sunset Boulevard",
    "Sunset Bulevard",
    "Sunset Blvd.",
    "sunset boulevard",
    "sunset bulevard",
    "west sunset boulevard",
    "sunset blvd"
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "tc_33",
    "content": [
      "Andrew Lloyd Webber, Baron Lloyd-Webber   (born 22 March 1948) is an English composer and impresario of musical theatre. \n\nSeveral of his musicals have run for more than a decade both in the West End and on Broadway. He has composed 13 musica ... [TRUNCATED] ... same name, composed the song \"Fields of Sun\". The actual song was never used on the show, nor was it available on the CD soundtrack that was released at the time. He was however still credited for the unused song in the show's opening titles."
    ]
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Read the content and answer the following question.

Content: {content}

Question: {question}

Keep your The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets trivia_qa \
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
    datasets=['trivia_qa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```