# MicroVQA


## Overview

MicroVQA is an expert-curated benchmark for multimodal reasoning in microscopy-based scientific research. It evaluates AI models' ability to understand and reason about microscopy images across various scientific domains.

## Task Description

- **Task Type**: Scientific Microscopy Visual Question Answering
- **Input**: Microscopy image(s) + scientific question with choices
- **Output**: Correct answer choice
- **Domain**: Microscopy, scientific research, medical imaging

## Key Features

- Expert-curated microscopy questions
- Multiple microscopy image types
- Tests scientific visual reasoning
- Medical and biological imaging focus
- Multiple-choice format with CoT prompting

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on test split
- Simple accuracy metric
- Uses Chain-of-Thought (CoT) prompting


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `micro_vqa` |
| **Dataset ID** | [evalscope/MicroVQA](https://modelscope.cn/datasets/evalscope/MicroVQA/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `Medical`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,042 |
| Prompt Length (Mean) | 884.16 chars |
| Prompt Length (Min/Max) | 414 / 1704 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,977 |
| Images per Sample | min: 1, max: 8, mean: 1.9 |
| Resolution Range | 78x70 - 2782x3533 |
| Formats | png |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "9bca1e27",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E. Think step by step before answering.\n\nA cryo-electron tom ... [TRUNCATED] ... Facilitation of mitochondrial biogenesis to support increased cell growth\nD) Influence on mitochondrial calcium buffering capacity to maintain cellular homeostasis\nE) Modification of mitochondrial lipid composition affecting membrane fluidity"
        },
        {
          "image": "[BASE64_IMAGE: png, ~188.3KB]"
        }
      ]
    }
  ],
  "choices": [
    "Regulation of mitochondrial metabolism to meet high ATP demand",
    "Enhancement of oxidative phosphorylation efficiency under varying energy demands",
    "Facilitation of mitochondrial biogenesis to support increased cell growth",
    "Influence on mitochondrial calcium buffering capacity to maintain cellular homeostasis",
    "Modification of mitochondrial lipid composition affecting membrane fluidity"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "key_question": 644,
    "key_image": 125,
    "correct_answer": "Regulation of mitochondrial metabolism to meet high ATP demand",
    "question_0": "The mitochondria in the tomographic slice are shown in dark blue, with calcium granules highlighted in yellow. How might the presence of calcium granules within the mitochondria affect their function?",
    "answer_0": "The presence of calcium granules within mitochondria can regulate mitochondrial metabolism and energy production, potentially indicating a high demand for ATP in the neuron.",
    "comments_0": null,
    "incorrect_answer_0": "The calcium granules within the mitochondria lead to their structural destabilization, causing the mitochondria to fragment.",
    "question_1": "A cryo-electron tomography (cryo-ET) image displays primary *Drosophila melanogaster* neurons cultured on a micropatterned grid that directs cytoskeletal growth. In the tomographic slice, mitochondria are shown in dark blue, and calcium granules are highlighted in yellow. How might the presence of calcium granules within the mitochondria affect their function?",
    "choices_1": [
      "Inhibition of the electron transport chain, reducing ATP production",
      "Regulation of mitochondrial metabolism to meet high ATP demand",
      "Promotion of mitochondrial fission, leading to fragmented mitochondria",
      "Induction of apoptosis through the release of cytochrome c"
    ],
    "correct_index_1": 1,
    "question_2": "A cryo-electron tomography (cryo-ET) image showcases primary eukaryotic neurons cultured on a specialized substrate that guides cytoskeletal organization. In the tomographic slice, mitochondria appear in deep blue, and calcium granules are depicted in bright yellow. What potential impact could the accumulation of calcium granules within the mitochondria have on their function?",
    "choices_2": [
      "Facilitation of mitochondrial biogenesis to support increased cell growth",
      "Enhancement of oxidative phosphorylation efficiency under varying energy demands",
      "Regulation of mitochondrial metabolism to meet high ATP demand",
      "Modification of mitochondrial lipid composition affecting membrane fluidity",
      "Influence on mitochondrial calcium buffering capacity to maintain cellular homeostasis"
    ],
    "correct_index_2": 2,
    "question_3": "A cryo-electron tomography (cryo-ET) image showcases primary eukaryotic neurons cultured on a specialized substrate that guides cytoskeletal organization. In the tomographic slice, mitochondria appear in deep blue, and calcium granules are depicted in bright yellow. What potential impact could the accumulation of calcium granules within the mitochondria have on their function?",
    "choices_3": [
      "Facilitation of mitochondrial biogenesis to support increased cell growth",
      "Enhancement of oxidative phosphorylation efficiency under varying energy demands",
      "Regulation of mitochondrial metabolism to meet high ATP demand",
      "Modification of mitochondrial lipid composition affecting membrane fluidity",
      "Influence on mitochondrial calcium buffering capacity to maintain cellular homeostasis"
    ],
    "correct_index_3": 2,
    "task": 2,
    "task_str": "hypothesis_gen",
    "context_image_generation": "This image was generated using cryo-ET, where a 3 × 4 montage tomogram was reconstructed from primary Drosophila melanogaster neurons cultured on a micropatterned grid designed to guide cytoskeleton growth. The grid provides a straight-line p ... [TRUNCATED] ... he organization of cellular structures such as microtubules. The tomographic slice was then segmented, with specific cellular components like microtubules, mitochondria, and ribosomes identified and highlighted using color coding for clarity.",
    "context_motivation": "The motivation behind generating these images was to study the organization of the cytoskeleton and associated cellular structures in primary Drosophila melanogaster neurons. By culturing the neurons on a straight-line micropatterned grid, th ... [TRUNCATED] ... ly microtubules. This approach helps in understanding how physical cues from the environment can influence the spatial arrangement and interactions of cellular organelles, which is crucial for insights into cellular architecture and function.",
    "images_source": "https://www.nature.com/articles/s41592-023-02000-z",
    "image_caption": "The image shows a tomographic slice with a cross-section of a Drosophila melanogaster neuron with various cellular structures visible. The image is a fully stitched montage, providing a comprehensive view of the neuron's internal organization ... [TRUNCATED] ... distribution and alignment within the neuron, Mitochondria are depicted in dark blue, with yellow indicating the presence of calcium granules within them, Ribosomes are shown in light pink, with associated vesicles represented in darker cyan.",
    "key_person": 0
  }
}
```

*Note: Some content was truncated for display.*

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
    --datasets micro_vqa \
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
    datasets=['micro_vqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


