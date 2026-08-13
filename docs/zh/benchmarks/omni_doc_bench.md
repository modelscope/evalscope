# OmniDocBench


## 概述

此适配器保留了 EvalScope 原始的 981 页 OmniDocBench TSV 集成，以确保与现有评估结果兼容。

## 任务描述

- **任务类型**：文档解析与理解
- **输入**：PDF 页面图像
- **输出**：以 Markdown 格式表示的解析后文档结构
- **领域**：文档理解、OCR、版面分析

## 主要特性

- 使用旧版 `evalscope/OmniDocBench_tsv` 数据集，包含 981 个 PDF 页面
- 覆盖文本块、公式、表格和阅读顺序
- 保留现有的本地 Python 评分实现和指标名称不变
- 可用于复现现有的 EvalScope 评估结果

## 评估说明

- 此旧版 TSV 数据集未标记为特定的上游 OmniDocBench 发布版本。
- 对于新评估，请使用推荐的 `omni_doc_bench_v1_6` 基准测试。
- 实现了现有的 `end2end` 和 `quick_match` 评分路径。
- 指标：Edit_dist、BLEU、METEOR（文本）、TEDS（表格）
- 安装 `evalscope[omnidoc_bench]` 额外依赖项以支持旧版评分功能。
- 输出格式：包含 LaTeX 公式和 HTML 表格的 Markdown
- 此旧版集成产生的分数无法直接与 v1.6 版本的分数进行比较。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `omni_doc_bench` |
| **数据集ID** | [evalscope/OmniDocBench_tsv](https://modelscope.cn/datasets/evalscope/OmniDocBench_tsv/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `text_block`, `display_formula`, `table`, `reading_order`, `normalized_score` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 981 |
| 提示词长度（平均） | 1408 字符 |
| 提示词长度（最小/最大） | 1408 / 1408 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 981 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 516x729 - 10142x14342 |
| 格式 | jpeg |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "7c6fda98",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~321.8KB]"
        },
        {
          "text": " You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:\n\n    1. Text Processing:\n    - Accurately recognize all text content in the PDF image without guessing or i ... [TRUNCATED 924 chars] ... sible.\n\n    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.\n"
        }
      ]
    }
  ],
  "target": "{\"layout_dets\": [{\"category_type\": \"title\", \"poly\": [102.5999912116609, 120.87255879760278, 719.3118659856144, 120.87255879760278, 719.3118659856144, 194.14083813380114, 102.5999912116609, 194.14083813380114], \"ignore\": false, \"order\": 1, \"an ... [TRUNCATED 9876 chars] ... nguage\": \"simplified_chinese\", \"layout\": \"1andmore_column\", \"special_issue\": [\"watermark\"]}, \"page_no\": 11, \"height\": 1500, \"width\": 2667, \"image_path\": \"eastmoney_59cde7e939acc3124df9d3f2c85b5a0ec41b9da1157d5be38e098672022b47cb.pdf_11.jpg\"}}",
  "id": 0,
  "group_id": 0
}
```

*注：部分内容因显示需要已被截断。*

## 提示模板

**提示模板：**
```text
 You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

    1. Text Processing:
    - Accurately recognize all text content in the PDF image without guessing or inferring.
    - Convert the recognized text into Markdown format.
    - Maintain the original document structure, including headings, paragraphs, lists, etc.

    2. Mathematical Formula Processing:
    - Convert all mathematical formulas to LaTeX format.
    - Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
    - Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

    3. Table Processing:
    - Convert tables to HTML format.
    - Wrap the entire table with <table> and </table>.

    4. Figure Handling:
    - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

    5. Output Format:
    - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
    - For complex layouts, try to maintain the original document's structure and format as closely as possible.

    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.

```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `match_method` | `str` | `quick_match` | 评估时使用的评分匹配方法。可选值：['quick_match', 'simple_match', 'no_split'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets omni_doc_bench \
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
    datasets=['omni_doc_bench'],
    dataset_args={
        'omni_doc_bench': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
