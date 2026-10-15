# VisualOverload


## 概述

VisualOverload (CVPR 2026) 探索了在密集、高分辨率场景中的视觉理解能力。真实答案是私有的；官方评分使用基准测试的评估服务器。

## 任务描述

- **任务类型**: 视觉问答
- **输入**: 一幅高分辨率的画作和官方的 `default_prompt`
- **输出**: 根据问题的不同，输出一个选项字母、一个计数或一个OCR答案
- **领域**: 活动、属性、计数、OCR、推理和场景理解

## 主要特点

- 从ModelScope上的`evalscope/visualoverload`中获取2,720个问题，该数据集镜像自Hugging Face上的官方`paulgavrikov/visualoverload`
- 结合了多种选择题和自由形式的问题，难度分为简单、中等和困难
- 使用官方提示不变，并默认保留原始图像字节
- 以官方的`question_id`/`response`提交格式导出原始响应

## 评估说明

- 数据集从ModelScope镜像的公共`test`分割加载，进行零样本评估。
- `response_present` 是用于诊断非空响应的指标，而不是答案准确性。
- 官方分数仅从评估服务器获得；没有选择本地质量度量。
- 将模型报告目录中的`visualoverload_submission.json`上传到[官方评估器](https://huggingface.co/spaces/paulgavrikov/visualoverload-submit)。
- 需要登录Hugging Face才能进行官方评估；EvalScope不会自动提交。
- 每个问题生成一次（`repeats=1`）。导出时会拒绝重复的问题ID。
- 有限次运行会产生部分提交用于测试，而不是完整的基准测试分数。
- 模型处理器可能会调整或平铺图像；请报告这些设置。`max_image_bytes`可以显式启用压缩，这会改变视觉输入。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `visualoverload` |
| **数据集ID** | [evalscope/visualoverload](https://modelscope.cn/datasets/evalscope/visualoverload/summary) |
| **论文** | [论文](https://arxiv.org/abs/2509.25339) |
| **标签** | `MultiModal`, `QA`, `Reasoning` |
| **度量标准** | `response_present` |
| **默认样本数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,720 |
| 提示词长度（平均） | 116.59 字符 |
| 提示词长度（最小/最大） | 40 / 356 字符 |

**图像统计:**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 2,720 |
| 每个样本的图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 922x2500 - 3412x2432 |
| 格式 | jpeg |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "a4c42212",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~3.3MB]"
        },
        {
          "text": "Does the scene feature fantasy characters? Options:\nA. yes\nB. no\nAnswer with the option's letter from the given choices directly."
        }
      ]
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "4ba66530",
    "question_type": "choice",
    "category": "scene",
    "difficulty": "medium"
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets visualoverload \
    --limit 10  # Remove this line for formal evaluation
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['visualoverload'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
