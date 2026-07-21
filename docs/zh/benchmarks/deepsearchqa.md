# DeepSearchQA

## 概述

DeepSearchQA 是 Google DeepMind 推出的一项基准测试，用于评估深度研究智能体在开放网络上执行复杂多步骤信息检索任务的能力。该基准包含 900 个提示，涵盖 17 个领域，旨在衡量模型生成完整答案集合的能力，而不仅仅是单个答案的检索。

## 任务描述

- **任务类型**：搜索智能体事实型问答
- **输入**：自然语言形式的研究问题
- **输出**：单个答案或完整的答案集合（取决于问题）
- **评分方式**：使用大语言模型（LLM）作为评判器，通过语义匹配对比标准答案及答案类型

## 核心特性

- 测试从多个来源系统性整合碎片化信息的能力
- 对于答案为集合的问题，要求进行实体消歧和去重
- 同时惩罚检索不足和过度/幻觉生成的答案
- 使用 `problem_category` 提供分析元数据；推理过程中对模型隐藏 `answer_type`
- 兼容 EvalScope 智能体配置，支持原生或具备外部网络能力的智能体

## 智能体工具配置

DeepSearchQA 不硬编码指定搜索引擎。默认情况下，它通过 EvalScope 原生 AgentLoop 运行，不使用外部搜索工具。若要评估具备网络能力的智能体，请设置 `TaskConfig.agent_config` 并附加模型可用的搜索/获取工具。如果未指定 `NativeAgentConfig.max_steps`，DeepSearchQA 将使用其基准级 AgentLoop 默认值 30 步。

有关运行示例、MCP 搜索/获取配置和评估说明，请参阅 [DeepSearchQA 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/third_party/deepsearchqa.html)。

## 评估说明

- EvalScope 从 `eval` 切分中加载 ModelScope 数据集 `google/deepsearchqa`。
- 默认启用 LLM 评判器。官方起始代码使用 Gemini 2.5 Flash 模型配合 DeepSearchQA 评判提示，但 EvalScope 在本地运行时可使用任意已配置的评判模型。
- 主要指标为 `f1_score`；同时报告 `precision`、`recall` 以及空/无效响应率。
- `JudgeStrategy.RULE` 为冒烟测试提供保守的精确匹配/子串匹配回退机制，但不等同于官方的 LLM 评判。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `deepsearchqa` |
| **数据集ID** | [google/deepsearchqa](https://modelscope.cn/datasets/google/deepsearchqa/summary) |
| **论文** | [Paper](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf) |
| **标签** | `Agent`, `Knowledge`, `QA`, `Retrieval` |
| **指标** | `f1_score`, `precision`, `recall` |
| **默认示例数** | 0-shot |
| **评估切分** | `eval` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 900 |
| 提示词长度（平均） | 295.54 字符 |
| 提示词长度（最小/最大） | 49 / 1007 字符 |

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets deepsearchqa \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":30}' \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['deepsearchqa'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=30,
    ),
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
