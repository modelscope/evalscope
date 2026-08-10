# CC-OCR-V2


## Overview

CC-OCR V2 is a challenging OCR benchmark tailored to real-world enterprise document processing. It deliberately
over-samples the hard and corner cases that prior OCR benchmarks under-represent, such as photographed and
scanned tables, handwritten formulas, multi-page receipts, and low-quality multilingual scene text.

## Task Description

- **Task Type**: Text recognition, document parsing, document grounding, key information extraction, and document VQA
- **Input**: One or more document images plus the task instruction shipped with each sample
- **Output**: Free-form text, LaTeX, HTML tables, SMILES strings, JSON objects, or bounding boxes, depending on the track
- **Modalities**: Image + text, bilingual (Chinese / English) with 32 additional languages in the recognition track

## Key Features

- 7,093 official samples over 5 tracks and 16 sub-tasks, evaluated as one benchmark; 7,091 are
  loaded because the dataset repository ships no image for two of them
- **recognition**: multilingual (32 languages) and natural-scene text reading
- **parsing**: complex tables, general documents, handwritten formulas, molecular structures, and information boards
- **grounding**: text grounding (single box) and object grounding (multi-box detection with labels)
- **extraction**: schema-driven key information extraction over business, public-service, and regulated records
- **qa**: question answering over blueprints, dashboards, and financial documents
- Prompts come from the official dataset, so results stay comparable to the published leaderboard

## Evaluation Notes

- Every sample yields one `score` in `[0, 1]`; each track uses its official metric:
  recognition = token-level F1, parsing = edit similarity / TEDS, grounding = IoU,
  extraction = field-level F1, qa = substring match with ANLS fallback
- Subset scores are sample means; the per-track category also reports a macro average over its sub-tasks
- The grounding prompts ask for boxes on a 0-1000 grid; predictions in absolute pixels are rescaled
  as if normalized and therefore score close to zero, matching the official leaderboard behavior
- Full-page parsing targets are long, so allow a generous `max_tokens` (4096 or more)
- Requires: `apted`, `distance`, `lxml`, `python-Levenshtein`, `scipy`, `zss`
  (`pip install 'evalscope[cc_ocr_v2]'`)
- The dataset is a file tree of images and answers (about 5 GB). Only the tracks listed in
  `subset_list` are downloaded, so restricting subsets keeps the download small


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `cc_ocr_v2` |
| **Dataset ID** | [evalscope/CC-OCR-V2](https://modelscope.cn/datasets/evalscope/CC-OCR-V2/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2605.03903) |
| **Tags** | `Grounding`, `MultiLingual`, `MultiModal`, `QA` |
| **Metrics** | `score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 7,091 |
| Prompt Length (Mean) | 272.16 chars |
| Prompt Length (Min/Max) | 10 / 1330 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
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

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 7,116 |
| Images per Sample | min: 1, max: 3, mean: 1.0 |
| Resolution Range | 70x71 - 5313x7219 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `multi_lingual_recognition`

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

*Note: Some content was truncated for display.*

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cc_ocr_v2 \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

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
