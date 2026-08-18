# CountQA


## 概述

CountQA 探究物体计数这一基本感知能力，而多模态模型在此方面大多未经充分评估。该数据集的图像均在日常环境中手工拍摄，并刻意包含高密度物体、杂乱背景和遮挡情况，使得仅靠检测少量分离良好的物体无法完成计数任务。

## 任务描述

- **任务类型**：自由形式视觉问答（物体计数）
- **输入**：一张真实世界照片 + 一个计数问题（例如：“有多少件夹克？”）
- **输出**：一个整数
- **领域**：日常场景——食品杂货、厨房用具、工具、衣物、办公及户外物品

## 关键特性

- 包含 1,528 个问答对，覆盖 1,001 张图像；每张图像可能对应多个问题
- 真实计数值在拍摄过程中现场标注（而非事后标注），范围从 0 到 400
- 问题包括组合型问题，需对多种物体类型分别计数后求和
- 约一半图像属于杂乱场景而非聚焦于单一主体（每个样本元数据中标记为 ``is_focused``），场景类别记录在 ``categories`` 字段中

## 评估说明

- 默认评估使用 **test** 划分作为单一子集
- 主要指标：**Accuracy** (`accuracy`) —— 与真实整数值完全匹配
- 次要指标：**relaxed_acc** —— 论文中定义的宽松准确率，当预测值在真实值的 ±5% 范围内即视为正确
- 使用论文中的系统提示词原样不变；该提示词强制模型仅输出纯整数
- 答案解析规则：若回复本身是整数则直接采用；否则取其中第一个整数——此规则与论文中用于重写答案的大语言模型一致。若回复不含任何数字，则得分为 0，因此 `max_tokens` 必须为模型留出足够空间输出答案；若模型以叙述方式计数（如“第一行有 3 个……”），则以其提到的第一个数字评分，而非其最终陈述的总数
- 评分基于确定性算术规则，无需大语言模型裁判：请保持 `judge_strategy` 为 `rule` 或 `auto`，因为 `llm` 会用通用裁判分数替代上述两个指标。若需从忽略输出格式的模型回复中提取不同数字，可通过 `dataset_args` 添加运行时过滤器（例如 `filters={'regex': {'regex_pattern': '(\d+)', 'group_select': -1}}` 提取最后一个数字），而非修改适配器

- [论文](https://arxiv.org/abs/2508.06585)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `count_qa` |
| **数据集ID** | [evalscope/CountQA](https://modelscope.cn/datasets/evalscope/CountQA/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2508.06585) |
| **标签** | `MultiModal`, `QA`, `Reasoning` |
| **指标** | `accuracy`, `relaxed_acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,528 |
| 提示词长度（平均） | 467.52 字符 |
| 提示词长度（最小/最大） | 459 / 508 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 1,528 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 1714x178 - 2482x2500 |
| 格式 | jpeg, png |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "dd3eebdf",
      "content": "You are a helpful assistant that counts the number of items in an image. The user will provide an image and ask a question about the number of a certain type of item in the image. If the user question is referring to multiple objects, it means that you need to provide a sum of the number of items. You will count the number of items and return the number as an integer. Your output should STRICTLY be a single integer and nothing else."
    },
    {
      "id": "ba0d838c",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~202.2KB]"
        },
        {
          "text": "How many tiles are on the wall with the shower?"
        }
      ]
    }
  ],
  "target": "18",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "is_focused": false,
    "categories": [
      "Outdoor, Vehicles & Structural"
    ]
  }
}
```

## 提示模板

**系统提示词：**
```text
You are a helpful assistant that counts the number of items in an image. The user will provide an image and ask a question about the number of a certain type of item in the image. If the user question is referring to multiple objects, it means that you need to provide a sum of the number of items. You will count the number of items and return the number as an integer. Your output should STRICTLY be a single integer and nothing else.
```

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets count_qa \
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
    datasets=['count_qa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
