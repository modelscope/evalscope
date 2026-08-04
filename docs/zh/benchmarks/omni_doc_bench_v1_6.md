# OmniDocBench-v1.6


## 概述

OmniDocBench v1.6 用于评估端到端的文档解析能力，涵盖文本、公式、表格、版式和阅读顺序。本适配器严格限定使用官方 v1.6 版本的数据和评分协议。

## 任务描述

- **任务类型**：端到端文档解析
- **输入**：完整的文档页面图像
- **输出**：包含页面文本、公式、表格及阅读顺序的 Markdown 内容
- **领域**：多语言学术、金融、教科书、报纸、杂志和演示文稿类文档

## 主要特性

- 使用 `OpenDataLab/OmniDocBench` 数据集，并固定至 ModelScope 的特定版本 `297ee5063d6ecc36fe14f3eb4f456607cc895f4a`
- 包含 1,651 页：其中 1,355 页为基础页面，另有 100 页为公式难度高、99 页为版式难度高、97 页为表格难度高的页面
- 采用官方 v1.6 数据格式；不支持其他版本及旧版 TSV 格式
- 每页独立评分，使用官方 v1.6 评估器，在可复用的 ms-enclave Docker 沙箱中执行

## 评估说明

- 使用 MGAM `quick_match`、公式 CDM、表格 TEDS/TEDS-S、编辑距离以及阅读顺序评估方法。
- EvalScope 对每页指标取平均值，并仅基于聚合后的文本、公式和表格组件计算 Overall 分数。
- 编辑距离指标采用 0–1 量表。
- CDM、TEDS、TEDS-S 和 Overall 采用 0–100 量表。
- 需要支持 amd64 架构的 Docker 环境及 `evalscope[sandbox]` 依赖。
- 默认镜像已固定；允许覆盖自定义镜像，但不兼容的镜像将在评分阶段失败。
- 沙箱池默认使用一个容器；仅在内存充足时才建议增加 `sandbox.pool_size`。
- 官方镜像体积较大；评估前请确保有足够的磁盘空间和内存。
- 评分结果不可与旧版 `omni_doc_bench` 集成直接比较。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `omni_doc_bench_v1_6` |
| **数据集ID** | [OpenDataLab/OmniDocBench](https://modelscope.cn/datasets/OpenDataLab/OmniDocBench/summary) |
| **论文** | [Paper](https://github.com/opendatalab/OmniDocBench) |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `text_block_Edit_dist`, `display_formula_Edit_dist`, `display_formula_CDM`, `table_TEDS`, `table_TEDS_structure_only`, `table_Edit_dist`, `reading_order_Edit_dist`, `overall` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,651 |
| 提示词长度（均值） | 1408 字符 |
| 提示词长度（最小/最大） | 1408 / 1408 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 1,651 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 570x829 - 10142x14342 |
| 格式 | jpeg, png |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "a93e81da",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~433.3KB]"
        },
        {
          "text": " You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:\n\n    1. Text Processing:\n    - Accurately recognize all text content in the PDF image without guessing or i ... [TRUNCATED 924 chars] ... sible.\n\n    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.\n"
        }
      ]
    }
  ],
  "target": "{\"layout_dets\": [{\"category_type\": \"text_block\", \"poly\": [268.9431, 319.97520000000003, 322.9962, 319.97520000000003, 322.9962, 351.0839, 268.9431, 351.0839], \"ignore\": false, \"order\": 2, \"anno_id\": \"box_id_0\", \"attribute\": {}, \"text\": \"that\" ... [TRUNCATED 7763 chars] ... th\": 1653, \"image_path\": \"page-d1561665-5359-42fe-920c-d6e3bff81953.png\", \"page_attribute\": {\"data_source\": \"book\", \"language\": \"english\", \"layout\": \"single_column\", \"special_issue\": [], \"subset\": \"equation_hard\"}}, \"extra\": {\"relation\": []}}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "omnidocbench_version": "v1.6",
    "dataset_revision": "297ee5063d6ecc36fe14f3eb4f456607cc895f4a",
    "image_name": "page-d1561665-5359-42fe-920c-d6e3bff81953.png"
  }
}
```

*注：部分内容因展示需要已被截断。*

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

## 沙箱配置

此基准测试需要沙箱环境以执行代码。

```json
{
  "image": "ghcr.io/zeng-weijun/omnidocbench-eval@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617",
  "entrypoint": [],
  "command": [
    "sleep",
    "infinity"
  ],
  "platform": "linux/amd64",
  "working_dir": "/workspace",
  "network_enabled": false,
  "tools_config": {
    "python_executor": {}
  }
}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets omni_doc_bench_v1_6 \
    --sandbox '{"enabled": true}' \
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
    datasets=['omni_doc_bench_v1_6'],
    sandbox={'enabled': True},
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
