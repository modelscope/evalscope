# CC-OCR-V2


## 概述

CC-OCR V2 是一个面向真实世界企业文档处理的高难度 OCR 基准。它有意过度采样以往 OCR 基准中代表性不足的高难度和边缘情况，例如拍摄和扫描的表格、手写公式、多页收据以及低质量的多语言场景文本。

## 任务描述

- **任务类型**：文本识别、文档解析、文档定位、关键信息提取和文档 VQA
- **输入**：一张或多张文档图像，以及每个样本附带的具体任务指令
- **输出**：自由文本、LaTeX、HTML 表格、SMILES 字符串、JSON 对象或边界框，具体取决于赛道
- **模态**：图像 + 文本，双语（中文 / 英文），识别赛道包含另外 32 种语言

## 主要特点

- 5 个赛道、16 个子任务共 7,093 个样本，作为一个基准进行评估
- **recognition**：多语言（32 种语言）和自然场景文本读取
- **parsing**：复杂表格、通用文档、手写公式、分子结构和信息板
- **grounding**：文本定位（单框）和对象定位（带标签的多框检测）
- **extraction**：面向商业、公共服务和受监管记录的 schema 驱动关键信息提取
- **qa**：针对蓝图、仪表板和财务文档的问答
- 提示词来自官方数据集，因此结果可与公开排行榜进行比较

## 评估说明

- 每个样本产生一个 `[0, 1]` 范围内的 `score`；每个赛道使用其官方指标：
  recognition = `token-level F1`，parsing = 编辑相似度 / TEDS，grounding = IoU，
  extraction = 字段级 F1，qa = 子串匹配，并以 ANLS 作为回退
- 子集分数为样本均值；每个赛道类别还会报告其子任务的宏平均
- grounding 提示词要求边界框位于 0-1000 坐标网格上。如果模型返回的是绝对像素坐标，评分器仍会将其解释为 0-1000 坐标，并使用图像宽度和高度转换为像素。这可能导致边界框错位并显著降低 IoU 分数
- 整页解析目标较长，因此应允许较大的 `max_tokens`（4096 或更多）
- 依赖：`apted`、`distance`、`lxml`、`python-Levenshtein`、`scipy`、`zss`
  （`pip install 'evalscope[cc_ocr_v2]'`）
- 数据集是图像和答案的文件树（约 5 GB）。只会下载 `subset_list` 中列出的赛道，因此限制子集可使下载量较小
- 不完整的样本会被跳过并给出警告。完整运行包含 7,093 个样本；如果加载的样本较少，请在报告结果前检查快照和缺失文件警告。官方 Hugging Face 数据源可通过 `--dataset-hub huggingface` 获取


## 属性

| 属性 | 数值 |
|----------|-------|
| **基准测试名称** | `cc_ocr_v2` |
| **数据集ID** | [evalscope/CC-OCR-V2](https://modelscope.cn/datasets/evalscope/CC-OCR-V2/summary) |
| **论文** | [论文](https://arxiv.org/abs/2605.03903) |
| **标签** | `Grounding`、`MultiLingual`、`MultiModal`、`QA` |
| **评估指标** | `normalized_score` |
| **默认示例数** | 0-shot |
| **评测划分** | `test` |


## 数据统计

| 指标 | 数值 |
|--------|-------|
| 总样本数 | 7,093 |
| 提示词长度（平均） | 272.14 字符 |
| 提示词长度（最小/最大） | 10 / 1330 字符 |

**子集统计：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `multi_lingual_recognition` | 640 | 101 | 101 | 101 |
| `natural_scene_recognition` | 1,150 | 101.87 | 101 | 103 |
| `complex_table_parsing` | 300 | 327 | 327 | 327 |
| `formula_parsing` | 100 | 119 | 119 | 119 |
| `general_documents_parsing` | 300 | 258 | 258 | 258 |
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

| 指标 | 数值 |
|--------|-------|
| 图像总数 | 7,118 |
| 每样本图像数 | 最小: 1，最大: 3，平均: 1.0 |
| 分辨率范围 | 70x71 - 5313x7219 |
| 格式 | jpeg、png |


## 样例示例

**子集**：`multi_lingual_recognition`

```json
{
  "input": [
    {
      "id": "f971e55d",
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

*注意：部分显示内容已截断。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用命令行

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
