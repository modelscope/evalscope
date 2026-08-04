# OmniDocBench-v1.6


## 概述

OmniDocBench v1.6 评估端到端的文档解析能力，涵盖文本、公式、表格、版面布局和阅读顺序。此适配器严格限定于官方 v1.6 数据及评分协议。

## 版本与数据来源

- **基准测试**: `omni_doc_bench_v1_6`
- **数据集**: `OpenDataLab/OmniDocBench`，固定至 ModelScope 版本 `297ee5063d6ecc36fe14f3eb4f456607cc895f4a`
- **规模**: 共 1,651 页，包含 1,355 页的 v1.5 数据集以及 296 页公式、版面和表格相关的困难样本
- **兼容性**: 不支持其他 OmniDocBench 版本及旧版 TSV 集成

## 评估方式

每页由官方 v1.6 评估器在 ms-enclave Docker 沙箱中独立评分。沙箱复用固定的官方镜像，并运行 MGAM `quick_match`、公式 CDM、表格 TEDS/TEDS-S、编辑距离和阅读顺序评估。EvalScope 对官方页面指标取平均值，并在所有页面聚合完成后计算 Overall 分数。

- 编辑距离指标使用 0-1 尺度。
- CDM、TEDS、TEDS-S 和 Overall 使用 0-100 尺度。
- 需要支持 amd64 的 Docker 环境及 `evalscope[sandbox]`。
- 沙箱池默认使用一个容器；仅在内存充足时才应增加 `sandbox.pool_size`。
- 官方镜像体积较大；评估前请确保磁盘和内存空间充足。
- 评分结果不可直接与旧版 `omni_doc_bench` v1.5 集成进行比较。

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
| 提示词长度（平均） | 1408 字符 |
| 提示词长度（最小/最大） | 1408 / 1408 字符 |

**图像统计:**

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
      "id": "46712ccc",
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
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "omnidocbench_version": "v1.6",
    "dataset_revision": "297ee5063d6ecc36fe14f3eb4f456607cc895f4a",
    "annotation_sha256": "a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496",
    "image_name": "page-d1561665-5359-42fe-920c-d6e3bff81953.png",
    "annotation": {
      "layout_dets": [
        {
          "category_type": "text_block",
          "poly": [
            268.9431,
            319.97520000000003,
            322.9962,
            319.97520000000003,
            322.9962,
            351.0839,
            268.9431,
            351.0839
          ],
          "ignore": false,
          "order": 2,
          "anno_id": "box_id_0",
          "attribute": {},
          "text": "that"
        },
        {
          "category_type": "equation_isolated",
          "poly": [
            404.98500000000007,
            362.07719999999995,
            816.9871826171875,
            362.07719999999995,
            816.9871826171875,
            448.3244323730468,
            404.98500000000007,
            448.3244323730468
          ],
          "ignore": false,
          "order": 3,
          "anno_id": "box_id_1",
          "attribute": {},
          "latex": "$$AB = \n\\left[\\begin{array}{ccc}\n2 & 3 \\\\\n1 & 4\n\\end{array}\\right]\n\\left[\\begin{array}{ccc}\n5 & 2 & 1 \\\\\n3 & 8 & 6\n\\end{array}\\right]$$"
        },
        {
          "category_type": "text_block",
          "poly": [
            271.9185,
            551.0684,
            1379.9243999999999,
            551.0684,
            1379.9243999999999,
            698.8932,
            271.9185,
            698.8932
          ],
          "ignore": false,
          "order": 5,
          "anno_id": "box_id_2",
          "attribute": {},
          "text": "When an attempt is made to form the product $ \\mathbf{{BA}} $ , we discover that the dimensions are not compatible in this order because the rows of $ \\mathbf{B} $ are three-dimensional vectors and the columns of $ \\mathbf{A} $ are two-dimensional vectors. Hence the dot product of the $ j $ th row of and the $ k $ th column of $ \\mathbf{A} $ is not defined.$\\blacksquare$"
        },
        {
          "category_type": "text_block",
          "poly": [
            274.0674,
            718.073,
            1374.9654,
            718.073,
            1374.9654,
            795.0260999999999,
            274.0674,
            795.0260999999999
          ],
          "ignore": false,
          "order": 6,
          "anno_id": "box_id_3",
          "attribute": {},
          "text": "If it happens that $ \\mathbf{{AB}} = \\mathbf{{BA}} $ , we say that $ \\mathbf{A} $ and $ \\mathbf{B} $ commute. Most often,even when $ \\mathbf{{AB}} $ and $ \\mathbf{{BA}} $ are both defined, the products are not necessarily the same."
        },
        {
          "category_type": "text_block",
          "poly": [
            277.0428,
            799.0024000000001,
            1379.9243999999999,
            799.0024000000001,
            1379.9243999999999,
            1025.8854,
            277.0428,
            1025.8854
          ],
          "ignore": false,
          "order": 7,
          "anno_id": "box_id_4",
          "attribute": {},
          "text": "We now discuss how to use matrices to represent a linear system of equations. The linear equations in (3) can be written as a matrix product. The coefficients $a _ { k j }$ are stored in a matrix $\\pmb { A }$ (called the coefficient matrix) o ... [TRUNCATED 73 chars] ... atrix $\\pmb { X }$ of dimension $N \\times 1$ .The constants $\\boldsymbol { b } _ { k }$ are stored in a matrix $\\pmb { B }$ of dimension $M \\times 1$ . It is conventional to use column matrices for both $\\pmb { X }$ and $\\pmb { B }$ and write"
        },
        {
          "category_type": "equation_isolated",
          "poly": [
            284.78271484375,
            1038.919444522659,
            1277.9343000000001,
            1038.919444522659,
            1277.9343000000001,
            1328.019844522659,
            284.78271484375,
            1328.019844522659
          ],
          "ignore": false,
          "order": 8,
          "anno_id": "box_id_5",
          "attribute": {},
          "latex": "$$\\mathbf{A}\\mathbf{X} = \\left\\lbrack  \\begin{array}{cccccc} {a}_{11} & {a}_{12} & \\cdots & {a}_{1j} & \\cdots & {a}_{1N} \\\\  {a}_{21} & {a}_{22} & \\cdots & {a}_{2j} & \\cdots & {a}_{2N} \\\\  \\vdots & \\vdots & & \\vdots & & \\vdots \\\\  {a}_{k1} &  ... [TRUNCATED 221 chars] ... {1} \\\\  {x}_{2} \\\\  \\vdots \\\\  {x}_{j} \\\\  \\vdots \\\\  {x}_{N} \\end{array}\\right\\rbrack   = \\left\\lbrack  \\begin{array}{cccccc} {b}_{1} \\\\  {b}_{2} \\\\  \\vdots \\\\  {b}_{j} \\\\  \\vdots \\\\  {b}_{M} \\end{array}\\right\\rbrack   = \\mathbf{B}.\\tag{8}$$"
        },
        {
          "category_type": "text_block",
          "poly": [
            279.0264,
            1336.0368,
            1384.0569,
            1336.0368,
            1384.0569,
            1452.9868,
            279.0264,
            1452.9868
          ],
          "ignore": false,
          "order": 9,
          "anno_id": "box_id_6",
          "attribute": {},
          "text": "The matrix multiplication $ \\mathbf{{AX}} = \\mathbf{B} $ in (8) is reminiscent of the dot product for ordinary vectors, because each element ${b}_{k} $in $ \\mathbf{B} $ is the result obtained by taking the dot product of row $k$in matrix $ \\mathbf{A} $ with the column matrix $ \\mathbf{X} $ ."
        },
        {
          "category_type": "text_block",
          "poly": [
            281.01000000000005,
            1485.0311000000002,
            1385.0487,
            1485.0311000000002,
            1385.0487,
            1562.9198000000001,
            281.01000000000005,
            1562.9198000000001
          ],
          "ignore": false,
          "order": 10,
          "anno_id": "box_id_7",
          "attribute": {},
          "text": "Example 3.6. Express the system of linear equations (5) in Example 3.4 as a matrix product. Use matrix multiplication to verify that ${\\left\\lbrack \\begin{array}{lll} 4 & 3 & 3 \\end{array}\\right\\rbrack }^{\\prime } $ is the solution of (5):"
        },
        {
          "category_type": "equation_isolated",
          "poly": [
            278.9978942871094,
            1579.0589,
            1104.0387,
            1579.0589,
            1104.0387,
            1707.9378,
            278.9978942871094,
            1707.9378
          ],
          "ignore": false,
          "order": 11,
          "anno_id": "box_id_8",
          "attribute": {},
          "latex": "$$\\left[\\begin{array}{cccccc}\n0.125 & 0.200 & 0.400 \\\\\n0.375 & 0.500 & 0.600 \\\\\n0.500 & 0.300 & 0.000\\end{array}\\right]\n\\left[\\begin{array}{cccccc}\nx_1 \\\\\nx_2 \\\\\nx_3\\end{array}\\right]\n=\\left[\\begin{array}{cccccc}2.3 \\\\4.8 \\\\2.9\\end{array}\\right].\\tag{9}$$"
        },
        {
          "category_type": "text_block",
          "poly": [
            282.99359999999996,
            1725.0125,
            1386.0405,
            1725.0125,
            1386.0405,
            1815.9995999999999,
            282.99359999999996,
            1815.9995999999999
          ],
          "ignore": false,
          "order": 12,
          "anno_id": "box_id_9",
          "attribute": {},
          "text": "To verify that $ {\\left\\lbrack \\begin{array}{lll} 4 & 3 & 3 \\end{array}\\right\\rbrack }^{\\prime } $ is the solution of (5), we must show that $A{\\left\\lbrack \\begin{array}{lll} 4 & 3 & 3 \\end{array}\\right\\rbrack }^{\\prime } =  {\\left\\lbrack \\begin{array}{lll} {2.3} & {4.8} & {2.9} \\end{array}\\right\\rbrack }^{\\prime } $ :"
        },
        "... [TRUNCATED 6 more items] ..."
      ],
      "page_info": {
        "page_no": 0,
        "height": 2339,
        "width": 1653,
        "image_path": "page-d1561665-5359-42fe-920c-d6e3bff81953.png",
        "page_attribute": {
          "data_source": "book",
          "language": "english",
          "layout": "single_column",
          "special_issue": [],
          "subset": "equation_hard"
        }
      },
      "extra": {
        "relation": []
      }
    }
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板:**
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
