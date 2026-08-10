# PMC-VQA


## Overview

PMC-VQA is a large-scale medical visual question answering benchmark built from figures of
biomedical papers in the PubMed Central Open Access subset. This integration evaluates the
manually verified **test_clean** split, the 2,000-question subset the authors recommend for
reporting results.

## Task Description

- **Task Type**: Medical Visual Question Answering (single-answer multiple choice)
- **Input**: A biomedical figure plus a question with four candidate answers
- **Output**: A single answer letter (A/B/C/D)
- **Domain**: Medicine and biomedical imaging (radiology, pathology, microscopy, plus charts and diagrams found in papers)

## Key Features

- 2,000 questions over 1,440 distinct figures, each with exactly four answer options
- Questions were generated from figure captions and then manually verified, so test_clean is
  substantially cleaner than the raw 50k test split
- Covers a wide range of imaging modalities and diseases, as well as non-photographic figures
  such as plots and diagrams
- Requires reading fine-grained visual detail together with biomedical domain knowledge

## Evaluation Notes

- Primary metric: **Accuracy** over the four options
- Answers are extracted from the `ANSWER: [LETTER]` line requested by the prompt; the original
  paper instead matches free-form generations to the closest option string, which is only needed
  for models that cannot follow an answer format
- Keep `max_tokens` generous enough for the model to finish its answer line: when no
  `ANSWER:` line is present, the shared multiple-choice parser falls back to the last
  upper-case letter in the reply, so a truncated response may be scored as a lenient guess
- Images are shipped as a single `images.zip` (about 18 GB) in the dataset repository. It is
  downloaded once and the figures needed for the evaluated samples are read directly from the
  archive, so no extracted copy is kept on disk
- [Paper](https://arxiv.org/abs/2305.10415) | [GitHub](https://github.com/xiaoman-zhang/PMC-VQA)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `pmc_vqa` |
| **Dataset ID** | [evalscope/PMC-VQA](https://modelscope.cn/datasets/evalscope/PMC-VQA/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2305.10415) |
| **Tags** | `MCQ`, `Medical`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test_clean` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,000 |
| Prompt Length (Mean) | 343.61 chars |
| Prompt Length (Min/Max) | 241 / 1105 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,000 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 17x21 - 4130x3564 |
| Formats | jpeg |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "03f4a772",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~93.0KB]"
        },
        {
          "text": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nWhat is the name of the medical imaging technique used in this case?\n\nA) X-ray\nB) Magnetic resonance imaging\nC) Computed tomography\nD) Ultrasound"
        }
      ]
    }
  ],
  "choices": [
    "X-ray",
    "Magnetic resonance imaging",
    "Computed tomography",
    "Ultrasound"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "figure_path": "PMC8415802_FIG1.jpg"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets pmc_vqa \
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
    datasets=['pmc_vqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
