# BrowseComp


## 概述

BrowseComp 是 OpenAI 提出的浏览与搜索代理评测集，用于评估模型在需要持续检索、交叉验证和多跳推理时回答事实型问题的能力。该评测集包含 1,266 道难以直接查到的问题，答案通常较短且可以验证。EvalScope 默认从 ModelScope 镜像数据集 `evalscope/browse_comp` 加载数据。

## 任务描述

- **任务类型**：搜索代理事实型问答
- **输入**：通常需要持续网页浏览的高难度自然语言问题
- **输出**：解释、精确答案和置信度
- **评分方式**：LLM judge 将模型最终答案与参考答案进行比对

## 主要特点

- 评估模型的搜索持久性、检索策略和多跳证据整合能力
- 使用短答案，便于进行可控的答案判定
- 官方数据以加密 CSV 行发布，评测时根据每行 canary 解密
- 作为 Agent 类评测集归类，兼容 EvalScope agent loop 模式
- 默认支持单轮模型评测；提供 `TaskConfig.agent_config` 时支持 native/external agent 执行

## 评估说明

- 默认通过标准 EvalScope 数据集加载器从 ModelScope 加载 `evalscope/browse_comp`。
- 可以通过 `TaskConfig.agent_config` 使用 EvalScope agent loop 能力评测 BrowseComp，例如 native tool-use 或 external agent runner。
- 主指标为 `is_correct`，同时报告 `is_incorrect`。
- 默认启用 LLM judge；当使用 `JudgeStrategy.RULE` 时，会退化为标准化后的精确匹配。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `browsecomp` |
| **数据集 ID** | [evalscope/browse_comp](https://modelscope.cn/datasets/evalscope/browse_comp/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2504.12516) |
| **标签** | `Agent`, `Knowledge`, `QA` |
| **指标** | `is_correct`, `is_incorrect` |
| **默认示例数量（Shots）** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,266 |
| 提示词长度（平均） | 811.02 字符 |
| 提示词长度（最小/最大） | 424 / 2219 字符 |

## 样例示例

*样例不可用。BrowseComp 官方样本以加密形式发布，不在公开文档中展示解密后的题目或答案。*

## 提示模板

**提示模板：**
```text
{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets browsecomp \
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
    datasets=['browsecomp'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
