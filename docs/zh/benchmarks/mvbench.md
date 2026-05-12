# MVBench

## 概述

MVBench 是一个公开的视频理解评测集，用于评估多模态模型的时间感知、属性和状态推理、符号顺序理解以及高层认知能力。本原生适配器使用 Hugging Face 上的 `PKU-Alignment/MVBench` 镜像，该镜像提供原始标注和优化后的视频归档文件。

## 任务描述

- **任务类型**：视频多选问答
- **输入**：视频 + 问题 + 候选答案
- **输出**：单个答案字母
- **默认子集**：`action_antonym`，用于轻量 smoke test
- **完整评测**：20 个子集，每个子集 200 条样本

## 主要特性

- 使用公开评测集中的真实 MP4 视频输入
- 复用现有 `VisionLanguageAdapter` 和 `MultiChoiceAdapter` 流程
- 通过统一的 `dataset_hub` 抽象加载标注和视频归档
- 只按需解压当前样本实际引用的视频文件
- 通过 OpenAI 兼容的 `video_url` 内容路径发送本地视频
- 保留可选的 `start` / `end` / `fps` 视频元数据，并在样本有时间边界时自动加入片段提示
- 可通过 `dataset_args` 选择需要评测的 MVBench 子集

## 评估说明

- 默认配置使用 **0-shot** 评估
- 主要指标：**准确率（Accuracy）**
- 默认 `action_antonym` 子集下载较小的 `ssv2_video.zip` 归档，适合 CI 风格验证
- 完整 MVBench 评测需要按所选子集下载对应的公开视频归档
- 由于公开的 `PKU-Alignment/MVBench` 镜像托管在 Hugging Face，本适配器默认使用 Hugging Face；如果使用镜像数据集，可通过 `extra_params.dataset_hub` 和 `extra_params.dataset_id` 指定

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mvbench` |
| **数据集ID** | [PKU-Alignment/MVBench](https://huggingface.co/datasets/PKU-Alignment/MVBench) |
| **论文** | [MVBench](https://arxiv.org/abs/2311.17005) |
| **标签** | `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估数据划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,000 |
| 子集数量 | 20 |
| 每个子集样本数 | 200 |

## 支持的子集

`action_antonym`, `action_count`, `action_localization`, `action_prediction`, `action_sequence`, `character_order`, `counterfactual_inference`, `egocentric_navigation`, `episodic_reasoning`, `fine_grained_action`, `fine_grained_pose`, `moving_attribute`, `moving_count`, `moving_direction`, `object_existence`, `object_interaction`, `object_shuffle`, `scene_transition`, `state_change`, `unexpected_action`.

## 样例示例

```json
{
  "video": "166583.mp4",
  "question": "What is the action performed by the person in the video?",
  "candidates": [
    "Not sure",
    "Scattering something down",
    "Piling something up"
  ],
  "answer": "Piling something up"
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mvbench \
    --dataset-args '{"mvbench": {"subset_list": ["action_antonym"], "extra_params": {"dataset_hub": "huggingface"}}}' \
    --limit 10
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
            'subset_list': ['action_antonym'],
            'extra_params': {
                'dataset_hub': 'huggingface',
            },
        }
    },
    limit=10,
)

run_task(task_cfg=task_cfg)
```
