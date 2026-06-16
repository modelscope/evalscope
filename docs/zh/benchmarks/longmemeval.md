# LongMemEval


## 概述

LongMemEval 用于评估聊天助手的长期交互记忆能力。每个问题均基于带有时间戳的多轮会话用户-助手历史记录进行回答。

## 任务描述

- **任务类型**：长上下文 / 检索日志问答
- **输入**：多个带日期的聊天会话 + 当前问题日期 + 问题
- **输出**：基于历史记录的自由形式答案
- **子集**：`s`（约115K token的历史记录）、`m`（约500个会话，规模较大）和 `oracle`（仅包含证据会话）

## 主要特性

- 涵盖单会话、多会话、时序推理、知识更新、偏好及弃权类问题
- 支持完整历史记录的长上下文提示和官方检索日志提示
- 使用 LongMemEval 的 LLM 评判提示来评估语义答案正确性
- 仅从 ModelScope 下载所选的 JSON 文件

## 评估说明

- 默认子集为 `s`，并使用 `eval_mode=long_context` 进行标准长上下文评估
- 使用 `subset_list=['oracle']` 和 `extra_params.eval_mode='oracle_context'` 进行仅含证据的上限评估
- `m` 子集规模较大，必须显式请求
- `retrieval_log` 模式使用官方 LongMemEval 检索日志；其本身不执行嵌入检索

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `longmemeval` |
| **数据集ID** | [evalscope/longmemeval-cleaned](https://modelscope.cn/datasets/evalscope/longmemeval-cleaned/summary) |
| **论文** | N/A |
| **标签** | `LongContext`, `MultiTurn`, `QA`, `Retrieval` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 500 |
| 提示词长度（平均） | 515877.18 字符 |
| 提示词长度（最小/最大） | 480743 / 540467 字符 |

## 样例示例

**子集**: `s`

```json
{
  "input": [
    {
      "id": "99d5aff3",
      "content": "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to ... [TRUNCATED 513848 chars] ... you'll be able to optimize your pantry storage space, reduce clutter, and make meal prep and cooking more efficient. Happy organizing!\"}]\n\n\nCurrent Date: 2023/05/30 (Tue) 23:40\nQuestion: What degree did I graduate with?\nAnswer (step by step):"
    }
  ],
  "target": "Business Administration",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "e47becba",
    "question_type": "single-session-user",
    "question": "What degree did I graduate with?",
    "answer": "Business Administration",
    "question_date": "2023/05/30 (Tue) 23:40",
    "answer_session_ids": [
      "answer_280352e9"
    ],
    "is_abstention": false,
    "eval_mode": "long_context",
    "retrieved_ids": [
      "sharegpt_yywfIrx_0",
      "85a1be56_1",
      "sharegpt_Jcy1CVN_0",
      "sharegpt_Cr2tc1f_0",
      "sharegpt_DGTCD7D_0",
      "f6859b48_2",
      "52c34859_1",
      "ultrachat_231069",
      "sharegpt_qRdLQvN_7",
      "ultrachat_359984",
      "... [TRUNCATED 43 more items] ..."
    ]
  }
}
```

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `eval_mode` | `str` | `long_context` | 评估模式：oracle_context、long_context 或 retrieval_log。可选项：['oracle_context', 'long_context', 'retrieval_log'] |
| `retrieval_log_path` | `str | null` | `None` | eval_mode=retrieval_log 时使用的官方 LongMemEval 检索日志路径。 |
| `retriever_type` | `str` | `flat-session` | retrieval_log 模式下的检索提示格式。可选项：['flat-session', 'flat-turn'] |
| `history_format` | `str` | `json` | 历史记录渲染格式。可选项：['json', 'nl'] |
| `user_only` | `bool` | `False` | 是否仅保留历史记录中的用户发言。 |
| `reading_method` | `str` | `con` | 提示阅读方法。`con` 要求模型先提取信息并推理后再作答。可选项：['direct', 'con'] |
| `topk_context` | `int` | `1000` | 提示中包含的最大历史会话数或检索片段数。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets longmemeval \
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
    datasets=['longmemeval'],
    dataset_args={
        'longmemeval': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```