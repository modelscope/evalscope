# MMLU

The MMLU (Massive Multitask Language Understanding) benchmark is a comprehensive evaluation suite designed to assess the performance of language models across a wide range of subjects and tasks. It includes multiple-choice questions from various domains, such as history, science, mathematics, and more, providing a robust measure of a model's understanding and reasoning capabilities.

## Overview

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mmlu` |
| **Dataset ID** | [cais/mmlu](https://modelscope.cn/datasets/cais/mmlu/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |

## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 14,042 |
| Prompt Length (Mean) | 274.54 chars |
| Prompt Length (Min/Max) | 4 / 4671 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `abstract_algebra` | 100 | 123.77 | 41 | 243 |
| `anatomy` | 135 | 82.35 | 11 | 262 |
| `astronomy` | 152 | 86.02 | 19 | 261 |
| `business_ethics` | 100 | 157.77 | 17 | 422 |
| `clinical_knowledge` | 265 | 66.34 | 12 | 229 |
| `college_biology` | 144 | 137.9 | 37 | 434 |
| `college_chemistry` | 100 | 131.17 | 37 | 422 |
| `college_computer_science` | 100 | 288.5 | 33 | 1064 |
| `college_mathematics` | 100 | 167.04 | 10 | 421 |
| `college_medicine` | 173 | 305.34 | 14 | 4671 |
| `college_physics` | 102 | 198.04 | 50 | 438 |
| `computer_security` | 100 | 118.09 | 13 | 601 |
| `conceptual_physics` | 235 | 76.12 | 16 | 282 |
| `econometrics` | 114 | 208.29 | 22 | 711 |
| `electrical_engineering` | 145 | 74.75 | 13 | 226 |
| `elementary_mathematics` | 378 | 113.2 | 13 | 400 |
| `formal_logic` | 126 | 205.71 | 67 | 603 |
| `global_facts` | 100 | 112.23 | 34 | 397 |
| `high_school_biology` | 310 | 147.66 | 13 | 568 |
| `high_school_chemistry` | 203 | 129.39 | 21 | 580 |
| `high_school_computer_science` | 100 | 223.41 | 36 | 1068 |
| `high_school_european_history` | 165 | 1391 | 461 | 2593 |
| `high_school_geography` | 198 | 86.03 | 17 | 226 |
| `high_school_government_and_politics` | 193 | 96.57 | 29 | 399 |
| `high_school_macroeconomics` | 390 | 96.42 | 6 | 370 |
| `high_school_mathematics` | 270 | 149.73 | 17 | 647 |
| `high_school_microeconomics` | 238 | 100.55 | 14 | 429 |
| `high_school_physics` | 151 | 201.91 | 49 | 534 |
| `high_school_psychology` | 545 | 137.78 | 22 | 761 |
| `high_school_statistics` | 216 | 266.54 | 21 | 760 |
| `high_school_us_history` | 204 | 1204.44 | 400 | 2016 |
| `high_school_world_history` | 237 | 1341.99 | 373 | 3538 |
| `human_aging` | 223 | 77.71 | 8 | 211 |
| `human_sexuality` | 131 | 102.53 | 18 | 370 |
| `international_law` | 121 | 70.25 | 15 | 192 |
| `jurisprudence` | 108 | 109.39 | 32 | 317 |
| `logical_fallacies` | 163 | 103.18 | 19 | 596 |
| `machine_learning` | 112 | 174.86 | 16 | 544 |
| `management` | 103 | 70.26 | 19 | 192 |
| `marketing` | 234 | 139.5 | 18 | 403 |
| `medical_genetics` | 100 | 80.21 | 18 | 303 |
| `miscellaneous` | 783 | 86.66 | 18 | 739 |
| `moral_disputes` | 346 | 97.9 | 13 | 311 |
| `moral_scenarios` | 895 | 321.89 | 256 | 465 |
| `nutrition` | 306 | 91.28 | 4 | 403 |
| `philosophy` | 311 | 80.67 | 19 | 579 |
| `prehistory` | 324 | 89.74 | 13 | 250 |
| `professional_accounting` | 282 | 238.8 | 35 | 619 |
| `professional_law` | 1,534 | 830.42 | 34 | 2937 |
| `professional_medicine` | 272 | 653.81 | 247 | 1637 |
| `professional_psychology` | 612 | 146.29 | 23 | 759 |
| `public_relations` | 110 | 130.49 | 24 | 565 |
| `security_studies` | 245 | 86.57 | 16 | 232 |
| `sociology` | 201 | 76.13 | 15 | 532 |
| `us_foreign_policy` | 100 | 81.38 | 25 | 173 |
| `virology` | 166 | 89.04 | 17 | 833 |
| `world_religions` | 171 | 66.16 | 16 | 141 |

## Subsets

| Subset | Sample Count |
|--------|--------------|
| `abstract_algebra` | 100 |
| `anatomy` | 135 |
| `astronomy` | 152 |
| `business_ethics` | 100 |
| `clinical_knowledge` | 265 |
| `college_biology` | 144 |
| `college_chemistry` | 100 |
| `college_computer_science` | 100 |
| `college_mathematics` | 100 |
| `college_medicine` | 173 |
| `college_physics` | 102 |
| `computer_security` | 100 |
| `conceptual_physics` | 235 |
| `econometrics` | 114 |
| `electrical_engineering` | 145 |
| `elementary_mathematics` | 378 |
| `formal_logic` | 126 |
| `global_facts` | 100 |
| `high_school_biology` | 310 |
| `high_school_chemistry` | 203 |
| `high_school_computer_science` | 100 |
| `high_school_european_history` | 165 |
| `high_school_geography` | 198 |
| `high_school_government_and_politics` | 193 |
| `high_school_macroeconomics` | 390 |
| `high_school_mathematics` | 270 |
| `high_school_microeconomics` | 238 |
| `high_school_physics` | 151 |
| `high_school_psychology` | 545 |
| `high_school_statistics` | 216 |
| `high_school_us_history` | 204 |
| `high_school_world_history` | 237 |
| `human_aging` | 223 |
| `human_sexuality` | 131 |
| `international_law` | 121 |
| `jurisprudence` | 108 |
| `logical_fallacies` | 163 |
| `machine_learning` | 112 |
| `management` | 103 |
| `marketing` | 234 |
| `medical_genetics` | 100 |
| `miscellaneous` | 783 |
| `moral_disputes` | 346 |
| `moral_scenarios` | 895 |
| `nutrition` | 306 |
| `philosophy` | 311 |
| `prehistory` | 324 |
| `professional_accounting` | 282 |
| `professional_law` | 1,534 |
| `professional_medicine` | 272 |
| `professional_psychology` | 612 |
| `public_relations` | 110 |
| `security_studies` | 245 |
| `sociology` | 201 |
| `us_foreign_policy` | 100 |
| `virology` | 166 |
| `world_religions` | 171 |

## Sample Example

**Subset**: `abstract_algebra`

```json
{
  "input": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
  "choices": [
    "0",
    "4",
    "2",
    "6"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "abstract_algebra",
  "metadata": {
    "subject": "abstract_algebra"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## Usage

```python
from evalscope import run_task

results = run_task(
    model='your-model',
    datasets=['mmlu'],
)
```