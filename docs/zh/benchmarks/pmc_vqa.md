# PMC-VQA


## 概述

PMC-VQA 是一个大规模医学视觉问答基准数据集，构建自 PubMed Central 开放获取子集中生物医学论文的图表。本集成评估的是经过人工验证的 **test_clean** 划分，即作者推荐用于报告结果的 2,000 个问题子集。

## 任务描述

- **任务类型**：医学视觉问答（单答案多项选择）
- **输入**：一张生物医学图像以及一个包含四个候选答案的问题
- **输出**：单个答案字母（A/B/C/D）
- **领域**：医学与生物医学成像（包括放射学、病理学、显微镜图像，以及论文中常见的图表和示意图）

## 主要特点

- 包含 2,000 个问题，覆盖 1,440 张不同的图像，每个问题恰好有四个选项
- 问题由图像标题生成并经过人工验证，因此 test_clean 子集比原始的 5 万规模测试集干净得多
- 涵盖多种成像模态和疾病类型，还包括非摄影类图像，如图表和示意图
- 要求模型结合细粒度的视觉细节与生物医学领域知识进行推理

## 评估说明

- **主要指标**：四个选项上的 **准确率（Accuracy）**
- 答案从提示词要求的 `ANSWER: [LETTER]` 行中提取；原论文则将自由生成的答案匹配到最接近的选项字符串，仅适用于无法遵循指定答案格式的模型
- 设置足够大的 `max_tokens`，确保模型能完整输出答案行：若响应中未出现 `ANSWER:` 行，共享的多项选择解析器会回退到响应中的最后一个大写字母，因此截断的回复可能被宽松地计为猜测
- 图像以单个 `images.zip` 文件（约 18 GB）形式存放在数据集仓库中。该文件仅下载一次，评估时所需图像直接从压缩包中读取，不会在磁盘上保留解压后的副本
- [论文](https://arxiv.org/abs/2305.10415) | [GitHub](https://github.com/xiaoman-zhang/PMC-VQA)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `pmc_vqa` |
| **数据集ID** | [evalscope/PMC-VQA](https://modelscope.cn/datasets/evalscope/PMC-VQA/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2305.10415) |
| **标签** | `MCQ`, `Medical`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test_clean` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,000 |
| 提示词长度（平均） | 343.61 字符 |
| 提示词长度（最小/最大） | 241 / 1105 字符 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 2,000 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 17x21 - 4130x3564 |
| 格式 | jpeg |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "03f4a772",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~93.0KB]"
        },
        {
          "text": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nWhat is the name of the medical imaging technique used in this case?\n\nA) X-ray\nB) Magnetic resonance imaging\nC) Computed tomography\nD) Ultrasound"
        }
      ]
    }
  ],
  "choices": [
    "X-ray",
    "Magnetic resonance imaging",
    "Computed tomography",
    "Ultrasound"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "figure_path": "PMC8415802_FIG1.jpg"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets pmc_vqa \
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
    datasets=['pmc_vqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
