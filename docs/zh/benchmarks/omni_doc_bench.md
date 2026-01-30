# OmniDocBench

## 概述

OmniDocBench 是一个面向真实场景的多样化文档解析评测数据集，涵盖 9 种文档类型、4 种布局类型和 3 种语言类型，共包含 1,355 页 PDF 文档。

## 任务描述

- **任务类型**：文档解析与理解
- **输入**：PDF 页面图像
- **输出**：以 Markdown 格式表示的解析后文档结构
- **领域**：文档理解、OCR、版面分析

## 核心特性

- 覆盖 9 类文档类型的 1,355 页 PDF
- 丰富的标注信息：15 种块级元素类型和 4 种跨度级（span-level）元素类型
- 超过 2 万个块级标注和 8 万个跨度级标注
- 阅读顺序标注
- 覆盖范围包括：学术论文、财务报告、报纸、教科书、手写笔记等

## 评测说明

- 实现了官方 OmniDocBench-v1.5 中的 `end2end` 和 `quick_match` 方法
- 评测指标：Edit_dist、BLEU、METEOR（文本）、TEDS（表格）
- 依赖包：apted、distance、lxml、Polygon3、zss、rapidfuzz
- 输出格式：包含 LaTeX 公式和 HTML 表格的 Markdown

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `omni_doc_bench` |
| **数据集ID** | [evalscope/OmniDocBench_tsv](https://modelscope.cn/datasets/evalscope/OmniDocBench_tsv/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `text_block`, `display_formula`, `table`, `reading_order` |
| **默认示例数** | 0-shot |
| **评测划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 981 |
| 提示词长度（平均） | 1408 字符 |
| 提示词长度（最小/最大） | 1408 / 1408 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 981 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 516x729 - 10142x14342 |
| 图像格式 | jpeg |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "fa523475",
      "content": [
        {
          "text": " You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:\n\n    1. Text Processing:\n    - Accurately recognize all text content in the PDF image without guessing or i ... [TRUNCATED] ... sible.\n\n    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.\n"
        },
        {
          "image": "[BASE64_IMAGE: png, ~321.8KB]"
        }
      ]
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "layout_dets": [
      {
        "category_type": "title",
        "poly": [
          102.5999912116609,
          120.87255879760278,
          719.3118659856144,
          120.87255879760278,
          719.3118659856144,
          194.14083813380114,
          102.5999912116609,
          194.14083813380114
        ],
        "ignore": false,
        "order": 1,
        "anno_id": 6,
        "text": "国资背景基金情况",
        "line_with_spans": [
          {
            "category_type": "text_span",
            "poly": [
              109.3333333333331,
              121.73651418039208,
              722.1022134807848,
              121.73651418039208,
              722.1022134807848,
              195.75809149176507,
              109.3333333333331,
              195.75809149176507
            ],
            "text": "国资背景基金情况"
          }
        ],
        "attribute": {
          "text_language": "text_simplified_chinese",
          "text_background": "white",
          "text_rotate": "normal"
        }
      },
      {
        "category_type": "text_block",
        "poly": [
          97.71487020898245,
          226.92028692633914,
          1271.9932332148471,
          226.92028692633914,
          1271.9932332148471,
          264.88925750697814,
          97.71487020898245,
          264.88925750697814
        ],
        "ignore": false,
        "order": 2,
        "anno_id": 4,
        "text": "2022年备案基金规模小幅回升，但仍未恢复至资管新规出台前的水平",
        "line_with_spans": [
          {
            "category_type": "text_span",
            "poly": [
              99.66504579139392,
              227.6650457913944,
              1269.333333333333,
              227.6650457913944,
              1269.333333333333,
              271.3365750838786,
              99.66504579139392,
              271.3365750838786
            ],
            "text": "2022年备案基金规模小幅回升，但仍未恢复至资管新规出台前的水平"
          }
        ],
        "attribute": {
          "text_language": "text_simplified_chinese",
          "text_background": "white",
          "text_rotate": "normal"
        }
      },
      {
        "category_type": "figure_caption",
        "poly": [
          246.96994018554688,
          318.7444152832031,
          1088.26025390625,
          318.7444152832031,
          1088.26025390625,
          369.0964660644531,
          246.96994018554688,
          369.0964660644531
        ],
        "ignore": false,
        "order": 3,
        "anno_id": 3,
        "text": "2014年-2023Q3国资背景基金的备案数量及规模",
        "line_with_spans": [
          {
            "category_type": "text_span",
            "poly": [
              253.94664201855937,
              321.21295194692755,
              1076.1203813864063,
              321.21295194692755,
              1076.1203813864063,
              364.93470762745034,
              253.94664201855937,
              364.93470762745034
            ],
            "text": "2014年-2023Q3国资背景基金的备案数量及规模"
          }
        ],
        "attribute": {
          "text_language": "text_simplified_chinese",
          "text_background": "white",
          "text_rotate": "normal"
        }
      },
      {
        "category_type": "figure",
        "poly": [
          118.08102792118407,
          379.29373168945347,
          1299.4279383691976,
          379.29373168945347,
          1299.4279383691976,
          1028.2773128579047,
          118.08102792118407,
          1028.2773128579047
        ],
        "ignore": false,
        "order": 4,
        "anno_id": 2
      },
      {
        "category_type": "figure_caption",
        "poly": [
          1497.726318359375,
          318.7418518066406,
          2301.80224609375,
          318.7418518066406,
          2301.80224609375,
          367.1272888183594,
          1497.726318359375,
          367.1272888183594
        ],
        "ignore": false,
        "order": 5,
        "anno_id": 8,
        "text": "2014年-2023Q3国资背景基金数量TOP10地区",
        "line_with_spans": [
          {
            "category_type": "text_span",
            "poly": [
              1509.6758069519938,
              324.34247361866034,
              2292.4771492866826,
              324.34247361866034,
              2292.4771492866826,
              364.8196229053426,
              1509.6758069519938,
              364.8196229053426
            ],
            "text": "2014年-2023Q3国资背景基金数量TOP10地区"
          }
        ],
        "attribute": {
          "text_language": "text_simplified_chinese",
          "text_background": "white",
          "text_rotate": "normal"
        }
      },
      {
        "category_type": "figure",
        "poly": [
          1370.0374839590943,
          424.35013794251097,
          2552.3561471143494,
          424.35013794251097,
          2552.3561471143494,
          1026.8955618700252,
          1370.0374839590943,
          1026.8955618700252
        ],
        "ignore": false,
        "order": 6,
        "anno_id": 5
      },
      {
        "category_type": "title",
        "poly": [
          170.92340081387997,
          1069.7956822171332,
          326.21460986860313,
          1069.7956822171332,
          326.21460986860313,
          1111.7494049722532,
          170.92340081387997,
          1111.7494049722532
        ],
        "ignore": false,
        "order": 7,
        "anno_id": 9,
        "text": "核心发现",
        "line_with_spans": [
          {
            "category_type": "text_span",
            "poly": [
              169.67751098302242,
              1071.225836994341,
              328.08580770628134,
              1071.225836994341,
              328.08580770628134,
              1111.655822350311,
              169.67751098302242,
              1111.655822350311
            ],
            "text": "核心发现"
          }
        ],
        "attribute": {
          "text_language": "text_simplified_chinese",
          "text_background": "white",
          "text_rotate": "normal"
        }
      },
      {
        "category_type": "text_block",
        "poly": [
          172.66793877059249,
          1155.2640660519091,
          2514.2408071863138,
          1155.2640660519091,
          2514.2408071863138,
          1241.6284871157177,
          172.66793877059249,
          1241.6284871157177
        ],
        "ignore": false,
        "order": 8,
        "anno_id": 7,
        "text": "- 2018年4月资管新规出台后，国资背景基金备案数量增速放缓且规模骤减，受新冠疫情影响，2021年新增基金规模再次下降，虽然 2022年基金规模回升至1.25万亿元，但仍未恢复至资管新规出台前的水平，2023前三季度新增规模略低于2022年同期。",
        "line_with_spans": [
          {
            "category_type": "text_span",
            "poly": [
              165.603649650326,
              1150.009124125815,
              2509.333333333333,
              1150.009124125815,
              2509.333333333333,
              1198.666666666666,
              165.603649650326,
              1198.666666666666
            ],
            "text": "- 2018年4月资管新规出台后，国资背景基金备案数量增速放缓且规模骤减，受新冠疫情影响，2021年新增基金规模再次下降，虽然"
          },
          {
            "category_type": "text_span",
            "poly": [
              219.22996126565647,
              1201.1457902508969,
              2250.770752144285,
              1201.1457902508969,
              2250.770752144285,
              1243.9433217869077,
              219.22996126565647,
              1243.9433217869077
            ],
            "text": "2022年基金规模回升至1.25万亿元，但仍未恢复至资管新规出台前的水平，2023前三季度新增规模略低于2022年同期。"
          }
        ],
        "attribute": {
          "text_language": "text_simplified_chinese",
          "text_background": "white",
          "text_rotate": "normal"
        }
      },
      {
        "category_type": "text_block",
        "poly": [
          171.69999831539863,
          1278.820932742719,
          2512.084408886781,
          1278.820932742719,
          2512.084408886781,
          1365.690053585406,
          171.69999831539863,
          1365.690053585406
        ],
        "ignore": false,
        "order": 9,
        "anno_id": 1,
        "text": "- 截至2023Q3全国国资背景基金备案数量累计9196只，基金规模累计8.91万亿元。基金注册区域集中于广东省、浙江省和江苏省，广东省国资背景基金总规模遥遥领先。备案基金数量前10的省份基金数量占全国总量的 73% ，规模占全国总量的 68%。",
        "line_with_spans": [
          {
            "category_type": "text_span",
            "poly": [
              161.7899369148969,
              1278.308761376868,
              2508,
              1278.308761376868,
              2508,
              1317.333333333333,
              161.7899369148969,
              1317.333333333333
            ],
            "text": "- 截至2023Q3全国国资背景基金备案数量累计9196只，基金规模累计8.91万亿元。基金注册区域集中于广东省、浙江省和江苏省，广东"
          },
          {
            "category_type": "text_span",
            "poly": [
              222.66666666666688,
              1325.3333333333335,
              1623.8331583485456,
              1325.3333333333335,
              1623.8331583485456,
              1365.333333333333,
              222.66666666666688,
              1365.333333333333
            ],
            "text": "省国资背景基金总规模遥遥领先。备案基金数量前10的省份基金数量占全国总量的"
          },
          {
            "category_type": "equation_ignore",
            "poly": [
              1624.4165959289367,
              1327.0154193159506,
              1703.7259660435407,
              1327.0154193159506,
              1703.7259660435407,
              1363.1237504250385,
              1624.4165959289367,
              1363.1237504250385
            ],
            "text": "73%"
          },
          {
            "category_type": "text_span",
            "poly": [
              1704.6905743174548,
              1322.6134268787764,
              2053.985160092844,
              1322.6134268787764,
              2053.985160092844,
              1370.6736155849724,
              1704.6905743174548,
              1370.6736155849724
            ],
            "text": "，规模占全国总量的"
          },
          {
            "category_type": "equation_ignore",
            "poly": [
              2055.1374027302004,
              1326.3706276890023,
              2149.276980264608,
              1326.3706276890023,
              2149.276980264608,
              1365.7029169328305,
              2055.1374027302004,
              1365.7029169328305
            ],
            "text": "68%。"
          }
        ],
        "attribute": {
          "text_language": "text_simplified_chinese",
          "text_background": "white",
          "text_rotate": "normal"
        }
      },
      {
        "category_type": "abandon",
        "poly": [
          114.12910090860571,
          1403.1676953230935,
          175.21358196554792,
          1403.1676953230935,
          175.21358196554792,
          1462.6586681785502,
          114.12910090860571,
          1462.6586681785502
        ],
        "ignore": false,
        "order": null,
        "anno_id": 10
      },
      {
        "category_type": "footer",
        "poly": [
          180.18207532211585,
          1404.2778174322868,
          289.9793827860912,
          1404.2778174322868,
          289.9793827860912,
          1462.652231000048,
          180.18207532211585,
          1462.652231000048
        ],
        "ignore": false,
        "order": null,
        "anno_id": 0,
        "text": "CVINFO 投中信息",
        "line_with_spans": [
          {
            "category_type": "text_span",
            "poly": [
              178.18192276049803,
              1409.8767302579377,
              288.0868232114207,
              1409.8767302579377,
              288.0868232114207,
              1467.2607048296584,
              178.18192276049803,
              1467.2607048296584
            ],
            "text": "CVINFO 投中信息"
          }
        ],
        "attribute": {
          "text_language": "text_en_ch_mixed",
          "text_background": "white",
          "text_rotate": "normal"
        }
      }
    ],
    "extra": {
      "relation": [
        {
          "source_anno_id": 2,
          "target_anno_id": 3,
          "relation_type": "parent_son"
        },
        {
          "source_anno_id": 5,
          "target_anno_id": 8,
          "relation_type": "parent_son"
        }
      ]
    },
    "page_info": {
      "page_attribute": {
        "data_source": "PPT2PDF",
        "language": "simplified_chinese",
        "layout": "1andmore_column",
        "special_issue": [
          "watermark"
        ]
      },
      "page_no": 11,
      "height": 1500,
      "width": 2667,
      "image_path": "eastmoney_59cde7e939acc3124df9d3f2c85b5a0ec41b9da1157d5be38e098672022b47cb.pdf_11.jpg"
    }
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

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `match_method` | `str` | `quick_match` | 评测所用的匹配方法。可选项：['quick_match', 'simple_match', 'no_split'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets omni_doc_bench \
    --limit 10  # 正式评测时请移除此行
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
    limit=10,  # 正式评测时请移除此行
)

run_task(task_cfg=task_cfg)
```