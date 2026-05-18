# MVBench


## 概述

MVBench 是一个公开的多模态视频理解基准测试，涵盖时间感知、属性/状态推理、符号排序和高级认知任务。此原生适配器默认使用 ModelScope 上的 `PKU-Alignment/MVBench` 镜像，该镜像提供 JSON 标注文件及优化后的视频压缩包。

## 任务描述

- **任务类型**：视频多项选择题问答（Video multiple-choice question answering）
- **输入**：视频 + 问题 + 答案选项
- **输出**：单个正确答案字母
- **子集**：20 个 MVBench 任务；默认的冒烟测试子集为 `action_antonym`

## 评估说明

- 默认配置使用 **0-shot** 评估
- 主要指标：**准确率（Accuracy）**
- 默认的 `action_antonym` 子集会下载一个小型公开 MP4 压缩包用于快速验证
- 可通过设置 `subset_list` 参数指定额外的 MVBench 子集以进行完整基准测试
- 对于带时间范围的记录，保留起始/结束元数据，并在提示词中添加简短的片段指令

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mvbench` |
| **数据集ID** | [PKU-Alignment/MVBench](https://modelscope.cn/datasets/PKU-Alignment/MVBench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2311.17005) |
| **标签** | `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,000 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `action_antonym` | 200 | N/A | N/A | N/A |
| `action_count` | 200 | N/A | N/A | N/A |
| `action_localization` | 200 | N/A | N/A | N/A |
| `action_prediction` | 200 | N/A | N/A | N/A |
| `action_sequence` | 200 | N/A | N/A | N/A |
| `character_order` | 200 | N/A | N/A | N/A |
| `counterfactual_inference` | 200 | N/A | N/A | N/A |
| `egocentric_navigation` | 200 | N/A | N/A | N/A |
| `episodic_reasoning` | 200 | N/A | N/A | N/A |
| `fine_grained_action` | 200 | N/A | N/A | N/A |
| `fine_grained_pose` | 200 | N/A | N/A | N/A |
| `moving_attribute` | 200 | N/A | N/A | N/A |
| `moving_count` | 200 | N/A | N/A | N/A |
| `moving_direction` | 200 | N/A | N/A | N/A |
| `object_existence` | 200 | N/A | N/A | N/A |
| `object_interaction` | 200 | N/A | N/A | N/A |
| `object_shuffle` | 200 | N/A | N/A | N/A |
| `scene_transition` | 200 | N/A | N/A | N/A |
| `state_change` | 200 | N/A | N/A | N/A |
| `unexpected_action` | 200 | N/A | N/A | N/A |

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `dataset_id` | `str` | `PKU-Alignment/MVBench` | MVBench 标注和视频的数据集仓库 ID 或本地数据集根目录。 |
| `dataset_hub` | `str` | `modelscope` | 用于加载标注和视频压缩包的数据集平台。可选值：['huggingface', 'modelscope', 'local'] |
| `dataset_revision` | `str` | `` | 可选的数据集版本；留空则使用平台默认版本。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mvbench \
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
    datasets=['mvbench'],
    dataset_args={
        'mvbench': {
            # subset_list: ['action_antonym', 'action_count', 'action_localization']  # 可选，用于评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```