# CC-OCR-V2

## 概述

CC-OCR V2 是一个针对现实世界企业文档处理的具有挑战性的 OCR 基准测试。它特意对先前 OCR 基准测试中代表性不足的困难和边缘案例进行了过采样，例如拍摄和扫描的表格、手写公式、多页收据以及低质量的多语言场景文本。

## 任务描述

- **任务类型**：文本识别、文档解析、文档定位、关键信息提取和文档 VQA
- **输入**：一个或多个文档图像加上每个样本附带的任务指令
- **输出**：自由格式文本、LaTeX、HTML 表格、SMILES 字符串、JSON 对象或边界框，具体取决于赛道
- **模态**：图像 + 文本，双语（中文/英文），并在识别赛道中包含 32 种其他语言

## 主要特点

- 7,093 个样本分布在 5 个赛道和 16 个子任务中，作为一个基准测试进行评估
- **识别**：多语言（32 种语言）和自然场景文本阅读
- **解析**：复杂表格、通用文档、手写公式、分子结构和信息板
- **定位**：文本定位（单框）和对象定位（带标签的多框检测）
- **提取**：基于模式的关键信息提取，涵盖商业、公共服务和受监管记录
- **问答**：蓝图、仪表板和财务文档上的问答
- 提示来自官方数据集，因此结果与已发布的排行榜保持可比性

## 评估说明

- 每个样本产生一个 `[0, 1]` 范围内的 `score`；每个赛道使用其官方指标：
  识别 = token 级 F1，解析 = 编辑相似度 / TEDS，定位 = IoU，
  提取 = 字段级 F1，问答 = 子字符串匹配，如果失败则使用 ANLS 回退
- 子集分数是样本均值；每个赛道类别还报告了其子任务的宏平均值
- 定位提示要求在 0-1000 的网格上标注框；绝对像素的预测将被重新缩放
  仿佛被归一化，因此得分接近零，与官方排行榜行为一致
- 全页解析目标很长，因此允许较大的 `max_tokens`（4096 或更多）
- 需要：`apted`, `distance`, `lxml`, `python-Levenshtein`, `scipy`, `zss`
  (`pip install 'evalscope[cc_ocr_v2]'`)
- 数据集是一个包含图像和答案的文件树（约 5 GB）。只有列在
  `subset_list` 中的赛道会被下载，因此限制子集可以减少下载量
- 不完整的样本会带有警告跳过。完整运行包含 7,093 个样本；如果加载的样本较少，请在报告结果前检查快照和缺失文件警告。官方 Hugging Face
  来源可用 `--dataset-hub huggingface`

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `cc_ocr_v2` |
| **数据集ID** | [evalscope/CC-OCR-V2](https://modelscope.cn/datasets/evalscope/CC-OCR-V2/summary) |
| **论文** | [论文](https://arxiv.org/abs/2605.03903) |
| **标签** | `Grounding`, `MultiLingual`, `MultiModal`, `QA` |
| **指标** | `normalized_score` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 7,093 |
| 提示词长度（平均） | 272.14 字符 |
| 提示词长度（最小/最大） | 10 / 1330 字符 |

**各子集统计：**

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

| 指标 | 值 |
|--------|-------|
| 总图像数 | 7,118 |
| 每个样本的图像数 | 最小: 1, 最大: 3, 平均: 1.0 |
| 分辨率范围 | 70x71 - 5313x7219 |
| 格式 | jpeg, png |

## 样例示例

**子集**: `multi_lingual_recognition`

```json
{
  "input": [
    {
      "id": "8f1b7891",
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

*注：部分内容因显示原因被截断。*

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
    --limit 10  # Remove this line for formal evaluation
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
            # subset_list: ['multi_lingual_recognition', 'natural_scene_recognition', 'complex_table_parsing']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
