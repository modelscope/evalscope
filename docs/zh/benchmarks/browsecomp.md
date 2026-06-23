# BrowseComp


## 概述

BrowseComp 是 OpenAI 推出的一项用于评估浏览和搜索智能体的基准测试。它包含 1,266 个难以查找、以事实为导向的问题，每个问题都有简短且可验证的答案。EvalScope 从 ModelScope 加载该数据集的镜像版本（`evalscope/browse_comp`）。

## 任务描述

- **任务类型**：搜索智能体的事实型问答
- **输入**：具有挑战性的自然语言问题，通常需要持续的网页浏览才能解答
- **输出**：解释、精确答案和置信度
- **评分方式**：由大语言模型（LLM）裁判将最终答案与参考答案进行比对

## 主要特性

- 测试智能体的持久性、创造性搜索能力以及多跳证据收集能力
- 使用简短答案以简化评分过程
- 官方数据以加密 CSV 行的形式分发，并在评估时解密
- 被归类为智能体（Agent）基准测试，兼容 EvalScope 的智能体循环模式
- 默认支持单轮模型评估；当提供 `TaskConfig.agent_config` 时，也支持原生或外部智能体执行

## 评估说明

- 默认评估通过标准 EvalScope 数据集加载器从 ModelScope 加载 `evalscope/browse_comp`。
- 使用 `TaskConfig.agent_config` 可启用 EvalScope 智能体循环功能（如原生工具调用或外部智能体运行器）来评估 BrowseComp。
- 主要指标为 `is_correct`；同时也会报告 `is_incorrect`。
- 默认启用 LLM 裁判；若未启用，则回退到 `JudgeStrategy.RULE`，即归一化精确匹配。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `browsecomp` |
| **数据集 ID** | [evalscope/browse_comp](https://modelscope.cn/datasets/evalscope/browse_comp/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2504.12516) |
| **标签** | `Agent`, `Knowledge`, `QA` |
| **指标** | `is_correct`, `is_incorrect` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,266 |
| 提示词长度（平均） | 811.02 字符 |
| 提示词长度（最小/最大） | 424 / 2219 字符 |

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
{question}

你的回答应采用以下格式：
Explanation: {{你对最终答案的解释}}
Exact Answer: {{你简洁明确的最终答案}}
Confidence: {{你对答案的置信度，介于 0% 到 100% 之间}}
```

## 使用方法

### 使用 CLI

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