# olmOCR-Bench


## 概述

olmOCR-Bench 评估端到端的文档转录能力：模型读取一页渲染后的 PDF 页面，并返回该页面完整的 Markdown 转录结果，随后通过人工编写的单元测试（而非单一参考答案）进行验证。

## 任务描述

- **任务类型**：PDF 页面到 Markdown 的转录
- **输入**：一张渲染后的 PDF 页面图像
- **输出**：该页面完整的 Markdown 转录文本
- **领域**：学术论文、扫描书籍、历史文档扫描件及内部文件

## 核心特性

- 发布的基准数据包含 1,403 页 PDF 和 7,019 个单元测试；每个测试定义了正确转录必须满足的一项属性（如文本存在性、文本缺失性、阅读顺序、表格结构或基础合理性检查）
- 评分规则与官方 `olmocr` 基准实现完全一致（1:1 移植），因此各子集得分可直接与官方报告对比
- 本适配器覆盖五个非数学来源（`headers_footers`、`long_tiny_text`、`multi_column`、`old_scans`、`table_tests`），共 845 页和 3,634 个单元测试；两个纯数学来源（`arxiv_math`、`old_scans_math`）需进行 KaTeX 渲染公式比对，未包含在内
- 所有页面已预先渲染为 PNG 图像（长边为 2048 像素，与官方渲染器 `render_pdf_to_base64png` 使用 `target_longest_image_dim=2048` 的设置一致），并与单元测试一同打包为单个 parquet 文件托管于 ModelScope；适配器通过标准远程加载流程（一次下载，原生解析）加载数据，评估时无需进行 PDF 光栅化

## 评估说明

- 每个样本对应一页 PDF；其单元测试将针对模型转录结果进行评估，子集得分为通过测试的比例，与官方按来源计算的指标一致
- 主要指标为 `pass_rate`。每个子集得分等于官方对应来源的通过率（即该来源中通过的单元测试占比），因此子集得分与官方报告完全一致
- 官方总分为各来源通过率的无权重平均值，记录为报告中的 `macro_score`（按类别）；而 EvalScope 默认显示的总体得分为样本加权（按页面）的微平均值；除非各子集页面数量相等，否则两者不同，因此若需与官方报告对齐，请比较子集得分或 `macro_score`
- `num` 表示每个子集的 PDF 页面数（样本数），因此总样本数与预测记录一致，而非单元测试总数
- 模糊匹配阈值（`max_diffs`）、位置约束（`first_n`/`last_n`）、大小写敏感性及表格关系检查均严格遵循官方实现
- 空的 `null` 回复被视为空转录，与官方评测框架将 `natural_text=null` 存储为空文件的行为一致
- 需安装依赖：`pip install evalscope[olmocr_bench]`（包含 rapidfuzz、fuzzysearch、beautifulsoup4）；基准数据为 ModelScope 上的单个 parquet 文件（`evalscope/olmOCR-Bench`，含图像字节与单元测试），是 Hugging Face 版本的镜像，支持单次下载原生加载
- [论文](https://arxiv.org/abs/2502.18443) | [代码](https://github.com/allenai/olmocr) | [数据集](https://modelscope.cn/datasets/evalscope/olmOCR-Bench)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `olmocr_bench` |
| **数据集ID** | [evalscope/olmOCR-Bench](https://modelscope.cn/datasets/evalscope/olmOCR-Bench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2502.18443) |
| **标签** | `MultiModal`, `QA` |
| **指标** | `pass_rate` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 845 |
| 提示词长度（平均） | 599 字符 |
| 提示词长度（最小/最大） | 599 / 599 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `headers_footers` | 266 | 599 | 599 | 599 |
| `long_tiny_text` | 62 | 599 | 599 | 599 |
| `multi_column` | 231 | 599 | 599 | 599 |
| `old_scans` | 98 | 599 | 599 | 599 |
| `table_tests` | 188 | 599 | 599 | 599 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 845 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 2048x773 - 2048x1957 |
| 格式 | png |


## 样例示例

**子集**: `headers_footers`

```json
{
  "input": [
    {
      "id": "0c5d3fad",
      "content": [
        {
          "text": "Below is the image of one page of a PDF document. Just return the plain text representation of this document as if you were reading it naturally.\nTurn equations into a LaTeX representation, and tables into markdown format. Remove the headers  ... [TRUNCATED 115 chars] ... l in the document, so be sure to preserve any sentences that come from the previous page, or continue onto the next page, exactly as they are.\nIf there is no text at all that you think you should read, you can output null.\nDo not hallucinate."
        },
        {
          "image": "[BASE64_IMAGE: png, ~297.1KB]"
        }
      ]
    }
  ],
  "target": "headers_footers/0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed.pdf#page=1",
  "id": 0,
  "group_id": 0,
  "subset_key": "headers_footers",
  "metadata": {
    "pdf": "headers_footers/0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed.pdf",
    "page": 1,
    "tests": [
      {
        "pdf": "headers_footers/0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed.pdf",
        "page": 1,
        "id": "0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed_pg1_header_01",
        "type": "absent",
        "max_diffs": 2,
        "checked": "verified",
        "url": "https://webges.uv.es/uvTaeWeb/DescargarCertificadoPublicacion.do?codigo=ANUNCIO-C9-2022-1285",
        "text": "Certificado de publicación disponible en http://fandango.accv.es:8070/fa",
        "case_sensitive": false,
        "first_n": null,
        "last_n": null
      },
      {
        "pdf": "headers_footers/0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed.pdf",
        "page": 1,
        "id": "0058e04004009cc0df75aab998d3e107dc646b46_page_1_processed_pg1_header_02",
        "type": "absent",
        "max_diffs": 4,
        "checked": "verified",
        "url": "https://webges.uv.es/uvTaeWeb/DescargarCertificadoPublicacion.do?codigo=ANUNCIO-C9-2022-1285",
        "text": "Este documento será custodiado por la Agencia de Tecnología y Certificación Electrónica - ISTEC Pista de Ademuz S/N. 46100 Burjassot (Valencia). Tel. 902 482 481 Correo-e: accv@accv.es",
        "case_sensitive": false,
        "first_n": null,
        "last_n": null
      }
    ]
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Below is the image of one page of a PDF document. Just return the plain text representation of this document as if you were reading it naturally.
Turn equations into a LaTeX representation, and tables into markdown format. Remove the headers and footers, but keep references and footnotes.
Read any natural handwriting.
This is likely one page out of several in the document, so be sure to preserve any sentences that come from the previous page, or continue onto the next page, exactly as they are.
If there is no text at all that you think you should read, you can output null.
Do not hallucinate.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets olmocr_bench \
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
    datasets=['olmocr_bench'],
    dataset_args={
        'olmocr_bench': {
            # subset_list: ['headers_footers', 'long_tiny_text', 'multi_column']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
