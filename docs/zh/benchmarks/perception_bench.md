# PerceptionBench


## 概述

PerceptionBench 是由 Moonshot AI 提出的一项基准测试，用于评估多模态大语言模型（MLLM）的原子级视觉感知能力。该基准采用自底向上的构建方式：通过对前沿 MLLM 在 42 个现有基准上最早出现的失败点进行诊断，归纳出一个错误分类体系，其中感知分支定义了十种原子级感知能力。每个问题仅聚焦于单一能力，因此其难度来源于感知本身，而非推理或知识。

## 任务描述

- **任务类型**：视觉感知（开放式问答）
- **输入**：一张或多张图像与一个问题交错排列
- **输出**：自由格式的简短答案，且具有唯一确定的参考答案
- **领域**：涵盖十种原子级视觉感知能力

## 主要特性

- 包含 3,000 个经过验证的问题，覆盖十种原子级感知能力
- 其中 1,800 个问题（60%）是从源基准中归因失败案例分解出的原子子问题；1,200 个问题（40%）是在补充图像上全新编写的问题
- 子集对应十种 `error_category` 标签：视觉关系、计数、属性、深度与 3D 感知、定位、比较、细粒度识别、上下文整合、OCR 和感知相关幻觉
- 支持多图像问题：图像通过 `<|image_N|>` 占位符嵌入到问题中
- 对于带有 `hint`（坐标约定或图像尺寸）的样本，会以系统消息形式传递，与官方消息构建器保持一致

## 评估说明

- 默认评估使用 **train** 切分（3,000 个样本，单切分数据集）
- 主要指标：**准确率（Accuracy）**，报告整体及各能力维度的结果
- 评分遵循官方协议：使用 LLM 评判器，根据教师评分提示（teacher-grading prompt）将自由格式答案与参考答案对比，并对每个样本返回严格的 0/1 判定（`[reason]` / `[judge] True|False`）；论文中使用的 GPT-oss-120B 模型在 300 个样本的人工审核中与人类判断的一致性达 99.7%
- 空输出或生成失败的样本直接记为 0 分，不调用评判器
- 需要配置 `judge_model_args` 以指定 LLM 评判器
- 数据集中图像以 base64 data URI 形式嵌入（首次使用时下载约 1.6 GB）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `perception_bench` |
| **数据集ID** | [moonshotai/PerceptionBench](https://modelscope.cn/datasets/moonshotai/PerceptionBench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2607.24957) |
| **标签** | `MultiModal`, `QA` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估切分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,000 |
| 提示词长度（平均） | 233.87 字符 |
| 提示词长度（最小/最大） | 29 / 1076 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `visual_relation_error` | 330 | 275.62 | 43 | 876 |
| `visual_counting_error` | 330 | 161.11 | 37 | 831 |
| `visual_attribute_error` | 330 | 225.58 | 34 | 1006 |
| `depth_3d_perception_error` | 330 | 278.5 | 60 | 976 |
| `visual_localization_error` | 330 | 284.79 | 62 | 1076 |
| `visual_comparison_error` | 279 | 270.14 | 39 | 801 |
| `fine_grained_recognition_error` | 290 | 225.91 | 44 | 917 |
| `context_integration_error` | 255 | 277.04 | 58 | 845 |
| `ocr_error` | 255 | 175.39 | 29 | 934 |
| `hallucination` | 271 | 150.94 | 42 | 515 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 3,567 |
| 每样本图像数 | 最小: 1, 最大: 8, 平均: 1.19 |
| 分辨率范围 | 101x64 - 5712x4953 |
| 格式 | jpeg, png, webp |


## 样例示例

**子集**: `visual_relation_error`

```json
{
  "input": [
    {
      "id": "28bf28ec",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~97.9KB]"
        },
        {
          "text": "How many arrows does the dashed box intersect with? Just answer with the number."
        }
      ]
    }
  ],
  "target": "4",
  "id": 0,
  "group_id": 0,
  "subset_key": "visual_relation_error",
  "metadata": {
    "index": 5,
    "problem": "<|image_1|>How many arrows does the dashed box intersect with? Just answer with the number.",
    "error_category": "visual_relation_error",
    "source_bmk": "NA",
    "source_idx": null
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
    --datasets perception_bench \
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
    datasets=['perception_bench'],
    dataset_args={
        'perception_bench': {
            # subset_list: ['visual_relation_error', 'visual_counting_error', 'visual_attribute_error']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
