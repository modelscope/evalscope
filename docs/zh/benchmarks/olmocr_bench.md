# olmOCR-Bench


## 概述

olmOCR-Bench 评估端到端的文档转录能力：模型读取一张渲染后的 PDF 页面图像，返回该页完整的 Markdown 转录，随后不是与单一参考答案比对，而是用人工编写的单元测试逐条校验。

## 任务描述

- **任务类型**：PDF 页面转 Markdown 转录
- **输入**：一张渲染后的 PDF 页面图像
- **输出**：该页完整的 Markdown 转录
- **领域**：学术论文、扫描书籍、历史扫描件与内部文档

## 主要特点

- 官方发布共 1,403 个 PDF、7,010 条单元测试；每条测试描述正确转录必须满足的一项属性（文本存在、文本缺失、阅读顺序、表格结构或基线健全性检查）
- 评分规则从官方 `olmocr` bench 实现 1:1 移植，各子集得分可与官方报告直接对比
- 本适配器覆盖五个非数学来源（`headers_footers`、`long_tiny_text`、`multi_column`、`old_scans`、`table_tests`），共 845 页、3,634 条单元测试；两个纯数学来源（`arxiv_math`、`old_scans_math`）依赖 KaTeX 渲染的公式比对，暂不包含
- 页面使用 `pypdfium2` 以 150 DPI 渲染；每个样本对应一张图像

## 评估说明

- 每个样本对应一个 PDF 页面；该页的单元测试逐条对模型转录执行，子集得分为通过的单元测试占比，与官方按来源统计的口径一致
- 主指标为 `pass_rate`；各子集得分即官方按来源统计的通过率。总报告得分为跨子集全部单元测试的（按测试数加权）通过率，与官方总分（对各 JSONL 得分求未加权平均）略有差异；需与官方报告精确对齐时请按子集分数对比
- 模糊匹配阈值（`max_diffs`）、位置约束（`first_n`/`last_n`）、大小写敏感性与表格关系检查均严格遵循官方实现
- 模型回复裸 `null` 时按空转录处理，与官方 harness 将 `natural_text=null` 存为空文件的方式一致
- 需要安装 `pip install evalscope[olmocr_bench]`（rapidfuzz、fuzzysearch、beautifulsoup4、pypdfium2）；数据集从 Hugging Face 下载
- [论文](https://arxiv.org/abs/2502.18443) | [代码](https://github.com/allenai/olmocr) | [数据集](https://huggingface.co/datasets/allenai/olmOCR-bench)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `olmocr_bench` |
| **数据集ID** | [allenai/olmOCR-bench](https://modelscope.cn/datasets/allenai/olmOCR-bench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2502.18443) |
| **标签** | `MultiModal`, `QA` |
| **指标** | `pass_rate` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

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
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
