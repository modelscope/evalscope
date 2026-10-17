# OpenDecisions


## 概述

OpenDecisions：2352 个类型决策。仅用于评估，加载单个 ModelScope 数据文件或 Hugging Face 测试分割。

## 任务描述

- **任务类型**：类型文本决策
- **输入**：状态和一个带有命名选项或有序级别的问题
- **输出**：选择、是的概率或序数得分分布
- **领域**：文本理解和决策

## 主要特点

- 1178 个选择任务、586 个 Noul 任务和 588 个评分任务；多语言输入
- 数据集：[jgeuter/OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions)；数据采用 CC BY-SA 4.0 许可，并附带每项来源证明，评分器采用 MIT 许可
- 保留选项顺序、源 ID 和评分标准描述
- 支持 System One 和在其声明能力范围内的原生 AutoTrust API

## 评估说明

- 零样本确定性评分；黄金答案和来源证明永远不会进入请求
- 主要：能力；官方智能和校准使用固定版本 v1.0 的评分器，修订号为 `5026a13ad6e5a8606810a3a5c4e410d8d6638e71`
- 官方值以比率形式存储（显示为百分比）；在智能评估中，缺失的预测计为错误
- 普通概率指标仅使用有效预测。Noul 的阈值在官方协议中为 0.5
- 无效响应被排除在普通指标之外，并反映在执行覆盖率中
- 评分标准有 3-6 个级别。AutoTrust 固定六级服务无法评估完整数据集
- 限制/随机运行只对选定项目进行评分，不应与完整的排行榜运行进行比较

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `open_decisions` |
| **数据集ID** | [evalscope/OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions/summary) |
| **论文** | N/A |
| **标签** | `Decision` |
| **指标** | `capability` |
| **默认示例数** | 0-shot |
| **评估分割** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,352 |
| 提示词长度（平均） | 1178.54 字符 |
| 提示词长度（最小/最大） | 208 / 5407 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "a68dac2e",
      "content": "{\"state\": \"Here are three expense lines from my trip. Turn them into a JSON list with fields merchant, date, amount_usd.\\n- Blue Cafe, 2024-03-02, $14.50\\n- MetroCab, 2024-03-02, $23.00\\n- Hotel Lira, 2024-03-04, $189.90\", \"question\": {\"type\" ... [TRUNCATED 79 chars] ... .\", \"criteria\": {\"chitchat\": \"Casual small talk, no task.\", \"creative_writing\": \"Compose original text.\", \"extraction_json\": \"Pull given details into a structured JSON shape.\", \"data_analysis\": \"Compute statistics or trends over the data.\"}}}",
      "internal": {
        "systemone": {
          "kind": "choice",
          "state": "Here are three expense lines from my trip. Turn them into a JSON list with fields merchant, date, amount_usd.\n- Blue Cafe, 2024-03-02, $14.50\n- MetroCab, 2024-03-02, $23.00\n- Hotel Lira, 2024-03-04, $189.90",
          "instructions": "Classify the task type of this request for routing.",
          "criteria": {
            "chitchat": "Casual small talk, no task.",
            "creative_writing": "Compose original text.",
            "extraction_json": "Pull given details into a structured JSON shape.",
            "data_analysis": "Compute statistics or trends over the data."
          },
          "answer_prefix": "ANSWER: "
        }
      }
    }
  ],
  "target": "extraction_json",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "source_id": "jb16-llm_router-9a24706d03",
    "decision_gold": "extraction_json",
    "decision_kind": "choice",
    "source_group": null,
    "family": "llm_router",
    "tier": "easy",
    "language": "en",
    "continuous": false,
    "decision_labels": [
      "chitchat",
      "creative_writing",
      "extraction_json",
      "data_analysis"
    ],
    "fixed_classes": false
  }
}
```

*注：部分内容因显示需要被截断。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --eval-type systemone_api \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets open_decisions \
    --limit 10  # Remove this line for formal evaluation
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='systemone_api',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['open_decisions'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
