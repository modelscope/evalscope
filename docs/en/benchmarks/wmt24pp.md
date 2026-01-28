# WMT2024++


## Overview

WMT2024++ is a comprehensive machine translation benchmark based on the WMT 2024 news translation task. It supports 54 language pairs with English as the source language, enabling evaluation of translation quality across diverse target languages.

## Task Description

- **Task Type**: Machine Translation
- **Input**: Source text in English with translation prompt
- **Output**: Translated text in the target language
- **Language Pairs**: 54 pairs (English to 54 target languages)

## Key Features

- Extensive multilingual coverage (54 target languages)
- News domain text for real-world applicability
- Multiple evaluation metrics (BLEU, BERTScore, COMET)
- Standardized prompt template for consistent evaluation
- Supports batch scoring for efficiency

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Metrics: **BLEU**, **BERTScore** (XLM-RoBERTa), **COMET** (wmt22-comet-da)
- Evaluates on **test** split
- Language-specific normalization applied
- COMET metric requires `unbabel-comet` package
- Subsets represent individual language pairs (e.g., `en-zh_cn`, `en-de_de`)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `wmt24pp` |
| **Dataset ID** | [extraordinarylab/wmt24pp](https://modelscope.cn/datasets/extraordinarylab/wmt24pp/summary) |
| **Paper** | N/A |
| **Tags** | `MachineTranslation`, `MultiLingual` |
| **Metrics** | `bleu`, `bert_score`, `comet` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 52,800 |
| Prompt Length (Mean) | 265.45 chars |
| Prompt Length (Min/Max) | 71 / 1047 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `en-ar_eg` | 960 | 263.26 | 75 | 1039 |
| `en-ar_sa` | 960 | 263.26 | 75 | 1039 |
| `en-bg_bg` | 960 | 269.26 | 81 | 1045 |
| `en-bn_in` | 960 | 265.26 | 77 | 1041 |
| `en-ca_es` | 960 | 265.26 | 77 | 1041 |
| `en-cs_cz` | 960 | 261.26 | 73 | 1037 |
| `en-da_dk` | 960 | 263.26 | 75 | 1039 |
| `en-de_de` | 960 | 263.26 | 75 | 1039 |
| `en-el_gr` | 960 | 261.26 | 73 | 1037 |
| `en-es_mx` | 960 | 265.26 | 77 | 1041 |
| `en-et_ee` | 960 | 267.26 | 79 | 1043 |
| `en-fa_ir` | 960 | 261.26 | 73 | 1037 |
| `en-fi_fi` | 960 | 265.26 | 77 | 1041 |
| `en-fil_ph` | 960 | 267.26 | 79 | 1043 |
| `en-fr_ca` | 960 | 263.26 | 75 | 1039 |
| `en-fr_fr` | 960 | 263.26 | 75 | 1039 |
| `en-gu_in` | 960 | 267.26 | 79 | 1043 |
| `en-he_il` | 960 | 263.26 | 75 | 1039 |
| `en-hi_in` | 960 | 261.26 | 73 | 1037 |
| `en-hr_hr` | 960 | 267.26 | 79 | 1043 |
| `en-hu_hu` | 960 | 269.26 | 81 | 1045 |
| `en-id_id` | 960 | 271.26 | 83 | 1047 |
| `en-is_is` | 960 | 269.26 | 81 | 1045 |
| `en-it_it` | 960 | 265.26 | 77 | 1041 |
| `en-ja_jp` | 960 | 267.26 | 79 | 1043 |
| `en-kn_in` | 960 | 265.26 | 77 | 1041 |
| `en-ko_kr` | 960 | 263.26 | 75 | 1039 |
| `en-lt_lt` | 960 | 271.26 | 83 | 1047 |
| `en-lv_lv` | 960 | 265.26 | 77 | 1041 |
| `en-ml_in` | 960 | 269.26 | 81 | 1045 |
| `en-mr_in` | 960 | 265.26 | 77 | 1041 |
| `en-nl_nl` | 960 | 261.26 | 73 | 1037 |
| `en-no_no` | 960 | 269.26 | 81 | 1045 |
| `en-pa_in` | 960 | 265.26 | 77 | 1041 |
| `en-pl_pl` | 960 | 263.26 | 75 | 1039 |
| `en-pt_br` | 960 | 271.26 | 83 | 1047 |
| `en-pt_pt` | 960 | 271.26 | 83 | 1047 |
| `en-ro_ro` | 960 | 267.26 | 79 | 1043 |
| `en-ru_ru` | 960 | 265.26 | 77 | 1041 |
| `en-sk_sk` | 960 | 263.26 | 75 | 1039 |
| `en-sl_si` | 960 | 269.26 | 81 | 1045 |
| `en-sr_rs` | 960 | 265.26 | 77 | 1041 |
| `en-sv_se` | 960 | 265.26 | 77 | 1041 |
| `en-sw_ke` | 960 | 265.26 | 77 | 1041 |
| `en-sw_tz` | 960 | 265.26 | 77 | 1041 |
| `en-ta_in` | 960 | 261.26 | 73 | 1037 |
| `en-te_in` | 960 | 263.26 | 75 | 1039 |
| `en-th_th` | 960 | 259.26 | 71 | 1035 |
| `en-tr_tr` | 960 | 265.26 | 77 | 1041 |
| `en-uk_ua` | 960 | 269.26 | 81 | 1045 |
| `en-ur_pk` | 960 | 259.26 | 71 | 1035 |
| `en-vi_vn` | 960 | 271.26 | 83 | 1047 |
| `en-zh_cn` | 960 | 267.26 | 79 | 1043 |
| `en-zh_tw` | 960 | 267.26 | 79 | 1043 |
| `en-zu_za` | 960 | 259.26 | 71 | 1035 |

## Sample Example

**Subset**: `en-ar_eg`

```json
{
  "input": [
    {
      "id": "557f3aa1",
      "content": [
        {
          "text": "Translate the following english sentence into arabic:\n\nenglish: Siso's depictions of land, water center new gallery exhibition\narabic:"
        }
      ]
    }
  ],
  "target": "رسومات سيسو عن الأرض والمية في معرضه الجديد",
  "id": 0,
  "group_id": 0,
  "subset_key": "en-ar_eg",
  "metadata": {
    "source_text": "Siso's depictions of land, water center new gallery exhibition",
    "target_text": "رسومات سيسو عن الأرض والمية في معرضه الجديد",
    "source_language": "en",
    "target_language": "ar_eg"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Translate the following {source_language} sentence into {target_language}:

{source_language}: {source_text}
{target_language}:
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets wmt24pp \
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
    datasets=['wmt24pp'],
    dataset_args={
        'wmt24pp': {
            # subset_list: ['en-ar_eg', 'en-ar_sa', 'en-bg_bg']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


