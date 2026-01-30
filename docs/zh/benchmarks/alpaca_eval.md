# AlpacaEval2.0

## 概述

AlpacaEval 2.0 是一个用于评估指令遵循语言模型的框架，它使用大语言模型（LLM）作为裁判，将待测模型的输出与一个强基线模型进行比较，并提供反映人类偏好的胜率（win-rate）指标。

## 任务描述

- **任务类型**：指令遵循评估（成对比较）
- **输入**：用户指令/问题
- **输出**：模型响应，与 GPT-4 Turbo 基线进行比较
- **指标**：相对于基线模型的胜率

## 主要特性

- 支持自动标注，可扩展性强
- 与 GPT-4 Turbo 基线输出进行对比
- 与人类偏好高度相关
- 评估成本低
- 测试通用的指令遵循能力

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用 LLM 裁判（默认：gpt-4-1106-preview）
- 基线模型：gpt-4-turbo 的输出
- 报告胜率（win rate）指标
- 注意：目前不支持长度控制的胜率计算

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `alpaca_eval` |
| **数据集ID** | [AI-ModelScope/alpaca_eval](https://modelscope.cn/datasets/AI-ModelScope/alpaca_eval/summary) |
| **论文** | N/A |
| **标签** | `Arena`, `InstructionFollowing` |
| **指标** | `winrate` |
| **默认示例数** | 0-shot |
| **评估划分** | `eval` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 805 |
| 提示词长度（平均） | 164.92 字符 |
| 提示词长度（最小/最大） | 12 / 1917 字符 |

## 样例示例

**子集**: `alpaca_eval_gpt4_baseline`

```json
{
  "input": [
    {
      "id": "95236545",
      "content": "What are the names of some famous actors that started their careers on Broadway?"
    }
  ],
  "target": "Several famous actors started their careers on Broadway before making it big in film and television. Here are a few notable examples:\n\n1. Sarah Jessica Parker - Before she was Carrie Bradshaw on \"Sex and the City,\" Sarah Jessica Parker was a  ... [TRUNCATED] ... f the many performers who have transitioned from the Broadway stage to broader fame in the entertainment industry. Broadway often serves as a proving ground for talent, and many actors continue to return to the stage throughout their careers.",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "generator": "gpt4_1106_preview",
    "dataset": "helpful_base"
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets alpaca_eval \
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
    datasets=['alpaca_eval'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```