# CC-OCR-V2


## 概述

CC-OCR V2 是一个面向真实企业文档处理场景的高难度 OCR 基准测试。它刻意对先前 OCR 基准测试中代表性不足的困难和边缘案例进行了过采样，例如拍摄或扫描的表格、手写公式、多页收据以及低质量的多语言自然场景文本。

## 任务描述

- **任务类型**：文本识别、文档解析、文档定位（grounding）、关键信息抽取和文档问答（VQA）
- **输入**：一张或多张文档图像，以及每个样本附带的任务指令
- **输出**：自由格式文本、LaTeX、HTML 表格、SMILES 字符串、JSON 对象或边界框，具体取决于任务赛道
- **模态**：图像 + 文本，双语（中文 / 英文），识别赛道额外包含 32 种语言

## 核心特性

- 共包含 7,093 个官方样本，覆盖 5 个赛道和 16 个子任务，统一作为单一基准进行评估；由于数据集仓库中缺少其中两个样本的图像，实际加载 7,091 个样本
- **识别（recognition）**：多语言（32 种语言）及自然场景文本阅读
- **解析（parsing）**：复杂表格、通用文档、手写公式、分子结构和信息公告板
- **定位（grounding）**：文本定位（单框）和对象定位（带标签的多框检测）
- **抽取（extraction）**：基于模式（schema-driven）的关键信息抽取，涵盖商业、公共服务和监管记录
- **问答（qa）**：针对蓝图、仪表盘和财务文档的问答任务
- 提示词（prompts）直接来自官方数据集，确保结果可与已发布的排行榜进行比较

## 评估说明

- 每个样本产生一个 `[0, 1]` 范围内的 `score`；各赛道使用其官方指标：
  - 识别 = token 级别 F1
  - 解析 = 编辑相似度 / TEDS
  - 定位 = IoU
  - 抽取 = 字段级别 F1
  - 问答 = 子串匹配，辅以 ANLS 回退策略
- 子集得分是样本得分的均值；各赛道类别还会报告其子任务的宏平均分
- 定位任务的提示要求在 0–1000 的归一化坐标网格上返回边界框；若预测使用绝对像素坐标，则会被重新缩放为归一化形式，否则得分将接近零，与官方排行榜行为一致
- 全页解析的目标输出较长，因此需设置较大的 `max_tokens`（建议 4096 或更高）
- 依赖项：`apted`, `distance`, `lxml`, `python-Levenshtein`, `scipy`, `zss`
  （可通过 `pip install 'evalscope[cc_ocr_v2]'` 安装）
- 数据集以图像和答案组成的文件树形式提供（约 5 GB）。仅下载 `subset_list` 中指定的赛道，因此限制子集可显著减小下载体积

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `cc_ocr_v2` |
| **数据集ID** | [evalscope/CC-OCR-V2](https://modelscope.cn/datasets/evalscope/CC-OCR-V2/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2605.03903) |
| **标签** | `Grounding`, `MultiLingual`, `MultiModal`, `QA` |
| **指标** | `score` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 7,091 |
| 提示词长度（平均） | 272.16 字符 |
| 提示词长度（最小/最大） | 10 / 1330 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `multi_lingual_recognition` | 639 | 101 | 101 | 101 |
| `natural_scene_recognition` | 1,150 | 101.87 | 101 | 103 |
| `complex_table_parsing` | 300 | 327 | 327 | 327 |
| `formula_parsing` | 100 | 119 | 119 | 119 |
| `general_documents_parsing` | 299 | 258 | 258 | 258 |
| `info_board_parsing` | 26 | 701 | 701 | 701 |
| `molecular_parsing` | 100 | 232 | 232 | 232 |
| `object_grounding` | 734 | 491.83 | 306 | 1330 |
| `text_grounding` | 734 | 369.88 | 358 | 468 |
| `business_transactions` | 340 | 793.91 | 702 | 1105 |
| `public_services` | 369 | 735.45 | 687 | 902 |
| `regulated_records` | 300 | 798.84 | 722 | 898 |
| `blueprint_qa` | 100 | 25.4 | 10 | 69 |
| `dashboards_fact_qa` | 400 | 66.59 | 19 | 159 |
| `dashboards_numeric_qa` | 500 | 70.49 | 19 | 148 |
| `financial_documents_qa` | 1,000 | 41.77 | 14 | 115 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 7,116 |
| 每样本图像数 | 最小: 1, 最大: 3, 平均: 1.0 |
| 分辨率范围 | 70x71 - 5313x7219 |
| 格式 | jpeg, png |


## 样例示例

**子集**: `multi_lingual_recognition`

```json
{
  "input": [
    {
      "id": "253834d2",
      "content": [
        {
          "image": "~/.cache/modelscope/hub/datasets/evalscope/CC-OCR-V2/recognition/multi_lingual_recognition/images/multi_lan_ocr_Arabic_Arabic_20/0c780237abcb.jpg"
        },
        {
          "text": "Please output only the text content from the image without any additional descriptions or formatting."
        }
      ]
    }
  ],
  "target": "الآن بحق السماء يا دجونا، أكل طعامك المتحجر غير المطابقة ....Demonstrandum\n.للمواصفات واتركني بسلام .”قال دجونا بحزن وهو ينظر إلى الحقيبة الفارغة: “لقد ذهب كل شيء أنا هنا!” صرخ بصوت غالي مرح، وكتم إليري أنينًا آخر عندما رأى السيد دوفال يقفز“\n ... [TRUNCATED 1733 chars] ... حنة الحشد“\nكان هناك ضحكة خفيفة. كان الشخص الضعيف القلب الذي خاطبه المرافق شابًا زنجيًا\nقويًا، يرتدي ملابس بنية سيمفونية أنيقة، وقبعته القشية مبهرة على الكربون السخام\n!الموجود في جلده. ضحكت فتاة جميلة ملونة على ذراعه. “هيا يا عزيزتي، سوف نريهم",
  "id": 0,
  "group_id": 0,
  "subset_key": "multi_lingual_recognition",
  "metadata": {
    "id": "0c780237abcb",
    "task": "recognition",
    "sub_task": "multi_lingual_recognition",
    "scenario": "multi_lan_ocr_Arabic_Arabic_20",
    "image_paths": [
      "~/.cache/modelscope/hub/datasets/evalscope/CC-OCR-V2/recognition/multi_lingual_recognition/images/multi_lan_ocr_Arabic_Arabic_20/0c780237abcb.jpg"
    ]
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cc_ocr_v2 \
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
    datasets=['cc_ocr_v2'],
    dataset_args={
        'cc_ocr_v2': {
            # subset_list: ['multi_lingual_recognition', 'natural_scene_recognition', 'complex_table_parsing']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
