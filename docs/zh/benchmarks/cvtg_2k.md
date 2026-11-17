# CVTG-2K


## 概述

CVTG-2K 是一个包含 2,000 条提示词的基准测试，用于评估复杂的视觉文本生成能力（TextCrafter, arXiv:2503.23461）。这些提示词包含 2-5 个带引号的文本区域，生成的图像必须准确渲染这些内容；提示词按区域数量组织在 CVTG 和 CVTG-Style 两个子集下（后者标注了大小、颜色和字体等风格属性）。

## 任务描述

- **任务类型**：文本到图像生成评估（文本渲染）
- **输入**：包含带引号文本区域的文本提示词
- **输出**：生成的图像，通过单词准确率（Word Accuracy）和针对带引号文本的单词级归一化编辑距离（NED）进行评分
- **领域**：面向文本到图像模型的复杂视觉文本生成

## 主要特点

- 来自官方 TextCrafter 发布的 2,000 条提示词（CVTG 和 CVTG-Style 子集，每条包含 2-5 个文本区域）
- 针对每条提示词中的单引号片段计算单词准确率和单词级 NED，复刻了官方 TextCrafter 评估器及其语料库级别的聚合方式
- 每种语言的两个指标共享同一个 PaddleOCR 引擎

## 评估说明

- 真实标签（Ground-truth）单词为每条提示词中的单引号片段，与官方 TextCrafter 评估器保持一致；重复出现的真实标签单词不会被消耗
- 指标包括：`cvtg_word_accuracy`（成员单词匹配）和 `cvtg_ned`（单词级最佳匹配 Levenshtein 相似度，值越高越好），两者均按照官方评估器的做法进行语料库级别聚合（总分子 / 总分母）
- 默认加载完整的 2,000 条提示词的 ModelScope 镜像。CVTG 和 CVTG-Style 配置分别包含 1,002 和 998 个测试样本；原始数据源为 [dnkdnk/CVTG-2K](https://huggingface.co/datasets/dnkdnk/CVTG-2K)。
- 安装 PaddleOCR 2.x 和 PaddlePaddle 以支持 OCR 评分（例如 `paddleocr==2.10.0` 和 `paddlepaddle==2.6.2`）。如果缺少依赖项或 OCR 失败，受影响的指标将被排除而不是记为零分。对于有效图像但未识别出任何文本的情况，视为缺失文本。
- 图像可以使用本地扩散模型或兼容 OpenAI 的图像生成服务生成；请设置 `model_task='image_generation'` 和 `eval_type='text2image'`。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `cvtg_2k` |
| **数据集ID** | [evalscope/CVTG-2K](https://modelscope.cn/datasets/evalscope/CVTG-2K/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2503.23461) |
| **标签** | `TextToImage` |
| **指标** | `CVTGWordAccuracy`, `CVTGNED` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,000 |
| 提示词长度（平均） | 195.92 字符 |
| 提示词长度（最小/最大） | 67 / 462 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `CVTG` | 1,002 | 161.05 | 67 | 343 |
| `CVTG-Style` | 998 | 230.93 | 106 | 462 |

## 样例示例

**子集**: `CVTG`

```json
{
  "input": [
    {
      "id": "dd076101",
      "content": "A quiet bookstore with a sign that says 'READ'. A coffee cup on the table with the word 'MORNING'."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "prompt": "A quiet bookstore with a sign that says 'READ'. A coffee cup on the table with the word 'MORNING'.",
    "carrier_list": [
      "sign",
      "cup"
    ],
    "sentence_list": [
      "a sign that says 'READ'.",
      "a coffee cup on the table with the word 'MORNING'."
    ],
    "benchmark_type": "CVTG",
    "region_count": 2,
    "category": "CVTG"
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
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets cvtg_2k \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='text2image',
    model_task='image_generation',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['cvtg_2k'],
    dataset_args={
        'cvtg_2k': {
            # subset_list: ['CVTG', 'CVTG-Style']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
