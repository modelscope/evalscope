# KINA


## Overview

KINA (Knowledge Index of Noah's Ark) is a high-density multidisciplinary knowledge benchmark for evaluating whether large language models can solve expert-level questions across 261 fine-grained disciplines. It is the first benchmark to incorporate disciplinary representativeness as a core design principle.

## Task Description

- **Task Type**: Multiple-Choice Question Answering (MCQ)
- **Input**: A discipline-specific question with up to 10 lettered options (A–J)
- **Output**: A single correct answer letter (A–J)
- **Domains**: 261 disciplines spanning Agronomy, Medicine, Engineering, Humanities, Natural Sciences, and more

## Key Features

- 899 test questions covering 261 fine-grained disciplines
- Each question has a unique correct answer among up to 10 options (A–J)
- Includes per-option explanations for training / analysis (not shown to the model)
- Designed to test deep domain knowledge, not retrieval or commonsense reasoning
- Introduced at 2077AI with a focus on disciplinary representativeness

## Evaluation Notes

- Default evaluation uses the **test** split (899 samples)
- Primary metric: **Accuracy** (acc) — Pass@1 for single-inference mode
- 0-shot Chain-of-Thought (CoT) evaluation, answer extracted from ``ANSWER: [LETTER]`` marker
- Discipline metadata is stored per-sample and available in review output; no per-discipline subset grouping
- [GitHub](https://github.com/weihao1115/KINA-Benchmark)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `kina` |
| **Dataset ID** | [evalscope/KINA](https://modelscope.cn/datasets/evalscope/KINA/summary) |
| **Paper** | [Paper](https://www.2077ai.com/kina) |
| **Tags** | `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 899 |
| Prompt Length (Mean) | 3280.88 chars |
| Prompt Length (Min/Max) | 482 / 22536 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "4750dae8",
      "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E,F,G,H,I,J. Think step by step before answering.\n\nUnder con ... [TRUNCATED 1998 chars] ...  it more economically sustainable under concentrate-restricted conditions.\nJ) Choose barn-dried hay because its intact fiber structure significantly increases milk fat percentage, making it more suitable for producing high-fat dairy products."
    }
  ],
  "choices": [
    "Silage, because anaerobic fermentation preserves soluble carbohydrates, true protein, and vitamins effectively, resulting in higher metabolizable energy density, superior palatability, and greater dry matter intake(DMI), thereby helping sustain milk yield when dietary concentrate is limited.",
    "Barn-dried hay, as it promotes higher DMI, enabling adequate nutrient intake and improving nitrogen utilization efficiency despite its lower crude protein concentration.",
    "Both forages are functionally equivalent and can be substituted on an equal dry matter basis, as they are both classified as roughages and exert no significant differential effect on lactation performance.",
    "Barn-dried hay, owing to its physically effective fiber structure and high lignin content, which enhance rumination activity and mitigate the risk of subacute ruminal acidosis.",
    "Silage, due to its high moisture content, which reduces voluntary water consumption and contributes to on-farm water conservation.",
    "Choose barn-dried hay because it contains no moisture, has a high dry matter content, and is therefore more \"nutrient-concentrated\" than wet silage.",
    "Choose silage because it contains probiotics that can directly improve gut health in dairy cows and serve as a protein source to replace concentrate.",
    "Choose silage. In southern China's rainy climate, hay is prone to mold growth and aflatoxin contamination, whereas silage avoids this risk and ensures raw milk safety-particularly important when concentrate supply is limited and reliance on safe forage is critical.",
    "Choose silage because it can be produced locally(e.g., whole-plant corn), harvested and stored mechanically, and offers lower cost per unit of nutrient compared to purchased high-quality hay, making it more economically sustainable under concentrate-restricted conditions.",
    "Choose barn-dried hay because its intact fiber structure significantly increases milk fat percentage, making it more suitable for producing high-fat dairy products."
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "discipline": "Agronomy/Animal Husbandry/Animal Nutrition and Feed Science"
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

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets kina \
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
    datasets=['kina'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


