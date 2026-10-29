# MT-Bench


## 概述

MT-Bench 在具有挑战性的开放式多轮对话场景中评估聊天助手。该基准测试随 LLM-as-a-judge 研究一同提出，使用强大的评判模型来衡量响应质量，超越了传统的封闭式基准测试。

## 任务描述

- **任务类型**：两轮开放式对话生成，并采用 LLM-as-a-judge 进行评分
- **输入**：一个初始用户请求，以及一个依赖于首轮助手回复的用户后续提问
- **输出**：助手对两轮对话的回复，完整对话内容将保留用于第二轮评分
- **领域**：写作、角色扮演、信息抽取、推理、数学、编程、STEM（科学、技术、工程、数学）及人文学科

## 主要特性

- 包含 80 个由人工编写的对话，涵盖八个类别，每个类别包含 10 个两轮问题。
- 适配器在第二轮上下文中保留了生成的第一轮回答，与官方对话流程保持一致。
- 官方协议在未配置任务级最大长度时，使用按类别设定的温度参数和最多 1,024 个生成 token。
- ModelScope 镜像提供了官方参考答案，用于参考引导式的推理、数学和编程评分提示；其中缺失的一个编程参考答案来自官方 FastChat 评测器。

## 评测说明

- 官方 MT-Bench 协议使用 `gpt-4` 作为单答案评判模型，并报告所有有效第一轮和第二轮评分在 1-10 分制下的平均值。
- 如需获得与官方结果可比的运行结果，请通过 `judge.models` 配置 `gpt-4`。允许使用兼容的自定义评判模型，但其得分不得直接与官方 GPT-4 结果进行比较。
- 推理、数学和编程类别的评分提示包含数据集提供的参考答案；其他类别则使用通用质量评分标准，涵盖有用性、相关性、准确性、深度、创造性和细节。
- 评判模型的回复必须满足 EvalScope 的 JSON 输出规范。解析或传输失败的样本将被排除，而非赋予零分。
- 资源链接：[论文](https://arxiv.org/abs/2306.05685) |
  [官方实现](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) |
  [数据集](https://modelscope.cn/datasets/HuggingFaceH4/mt_bench_prompts)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mt_bench` |
| **数据集ID** | [HuggingFaceH4/mt_bench_prompts](https://modelscope.cn/datasets/HuggingFaceH4/mt_bench_prompts/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2306.05685) |
| **标签** | `Coding`, `InstructionFollowing`, `Math`, `MultiTurn`, `QA`, `Reasoning` |
| **指标** | `judge_score`, `first_turn_judge_score`, `second_turn_judge_score` |
| **默认示例数** | 0-shot |
| **评测划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 80 |
| 提示词长度（平均） | 299.55 字符 |
| 提示词长度（最小/最大） | 38 / 1642 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `writing` | 10 | 211.7 | 126 | 365 |
| `roleplay` | 10 | 306.5 | 140 | 511 |
| `reasoning` | 10 | 279.4 | 77 | 862 |
| `math` | 10 | 156.1 | 38 | 296 |
| `coding` | 10 | 162.1 | 69 | 541 |
| `extraction` | 10 | 959.5 | 385 | 1642 |
| `stem` | 10 | 202.9 | 92 | 319 |
| `humanities` | 10 | 118.2 | 68 | 219 |

## 样例示例

**子集**: `writing`

```json
{
  "input": [
    {
      "id": "52192365",
      "content": "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "writing",
  "metadata": {
    "category": "writing",
    "prompt_id": 44067482,
    "turns": [
      "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
      "Rewrite your previous response. Start every sentence with the letter A."
    ],
    "references": []
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mt_bench \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
    --limit 10  # 正式评测时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['mt_bench'],
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    dataset_args={
        'mt_bench': {
            # subset_list: ['writing', 'roleplay', 'reasoning']  # 可选，用于评测特定子集
        }
    },
    limit=10,  # 正式评测时请删除此行
)

run_task(task_cfg=task_cfg)
```
