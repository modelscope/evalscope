# InfoVQA


## Overview

InfoVQA (Infographic Visual Question Answering) is a benchmark designed to evaluate AI models' ability to answer questions based on information-dense images such as charts, graphs, diagrams, maps, and infographics. It focuses on understanding complex visual information presentations.

## Task Description

- **Task Type**: Infographic Question Answering
- **Input**: Infographic image + natural language question
- **Output**: Single word or phrase answer
- **Domains**: Data visualization, information graphics, visual reasoning

## Key Features

- Focuses on information-dense visual content
- Covers charts, graphs, diagrams, maps, and infographics
- Requires understanding visual layouts and data representations
- Tests information extraction and reasoning abilities
- Questions vary in complexity from direct lookup to inference

## Evaluation Notes

- Default evaluation uses the **validation** split
- Primary metric: **ANLS** (Average Normalized Levenshtein Similarity)
- Answers should be in format "ANSWER: [ANSWER]"
- Includes OCR text extraction as metadata for analysis
- Uses same dataset source as DocVQA (InfographicVQA subset)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `infovqa` |
| **Dataset ID** | [lmms-lab/DocVQA](https://modelscope.cn/datasets/lmms-lab/DocVQA/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `anls` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,801 |
| Prompt Length (Mean) | 273.38 chars |
| Prompt Length (Min/Max) | 222 / 390 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,801 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 600x340 - 6250x9375 |
| Formats | jpeg |


## Sample Example

**Subset**: `InfographicVQA`

```json
{
  "input": [
    {
      "id": "03ab3147",
      "content": [
        {
          "text": "Answer the question according to the image using a single word or phrase.\nWhich social platform has heavy female audience?\nThe last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the question."
        },
        {
          "image": "[BASE64_IMAGE: png, ~249.7KB]"
        }
      ]
    }
  ],
  "target": "[\"pinterest\"]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "questionId": "98313",
    "answer_type": [
      "single span"
    ],
    "image_url": "https://blogs.constantcontact.com/wp-content/uploads/2019/03/Social-Media-Infographic.png",
    "ocr": "['{\"PAGE\": [{\"BlockType\": \"PAGE\", \"Geometry\": {\"BoundingBox\": {\"Width\": 0.9994840025901794, \"Height\": 0.9997748732566833, \"Left\": 0.0, \"Top\": 0.0}, \"Polygon\": [{\"X\": 0.0, \"Y\": 0.0}, {\"X\": 0.9994840025901794, \"Y\": 0.0}, {\"X\": 0.999484002590179 ... [TRUNCATED] ... 184143, \"Y\": 0.9778721332550049}, {\"X\": 0.5701684951782227, \"Y\": 0.9778721332550049}, {\"X\": 0.5701684951782227, \"Y\": 0.9896419048309326}, {\"X\": 0.47732439637184143, \"Y\": 0.9896419048309326}]}, \"Id\": \"43af6e92-c2ef-483c-b947-8b2d2073d756\"}]}']"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the question according to the image using a single word or phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the question.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets infovqa \
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
    datasets=['infovqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


