# ScreenSpot-Pro


## Overview

ScreenSpot-Pro is a GUI grounding benchmark built from authentic high-resolution screenshots of professional desktop software. Given a natural-language instruction, a model must locate the target UI element on the screen, which stresses fine-grained localization on large, densely populated displays.

## Task Description

- **Task Type**: GUI grounding (single click-point prediction)
- **Input**: A full-resolution desktop screenshot + an English instruction describing the target UI element
- **Output**: One click point `[x, y]` normalized to the range 0 to 1, given after an `Answer:` marker
- **Domain**: Professional desktop applications across CAD, Creative, Dev, Office, OS and Scientific software

## Key Features

- 1,581 expert-annotated instructions over 26 applications and 3 platforms (Windows, macOS, Linux)
- Screenshots are genuinely high-resolution (up to 6016x3384), so target elements often occupy well under 0.1% of the image
- Samples are grouped into six professional domains (`CAD`, `Creative`, `Dev`, `OS`, `Office`, `Scientific`), each exposed as a subset
- Every element is labelled as `text` or `icon`, enabling separate reporting for textual versus iconographic targets
- Ground-truth boxes are pixel coordinates paired with the original image size, and are normalized before scoring

## Evaluation Notes

- Primary metric: **accuracy** — a prediction is correct when the predicted point falls inside the ground-truth bounding box
- Secondary metrics: **text_acc** and **icon_acc**, each averaged over the samples of the corresponding `ui_type`
- Predictions are read from the answer line that the prompt requires (`Answer: [x, y]`), so reasoning traces cannot be mistaken for the answer. Replies ignoring the format fall back to scanning for unambiguous point notation only (`[x, y]` pairs or `<bbox>` tags); loose notation such as `x=.., y=..` and bare numbers is accepted only on the answer line, because in free prose it harvests layout bounds and ordinals instead of a click point
- A reply truncated before its answer line yields no prediction and scores 0 rather than a coordinate invented from its reasoning, so allow enough `max_tokens` for the model to finish answering
- Ground truth is normalized to [0, 1], and predictions are mapped into the same space by magnitude: values in [0, 1] are taken as normalized, values up to 1000 as the thousandths grid many VLMs emit, and larger values as pixels of the image the model received (every screenshot is at least 1920 px wide, so genuine pixel answers are classified correctly)
- The dataset ships a single `train` split, which is used as the evaluation split
- Images are large; `max_image_bytes` in `dataset_args` can cap the request size, and pixel-space predictions are normalized with the size of the image actually sent
- [Paper](https://arxiv.org/abs/2504.07981) | [GitHub](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `screenspot_pro` |
| **Dataset ID** | [lmms-lab/ScreenSpot-Pro](https://modelscope.cn/datasets/lmms-lab/ScreenSpot-Pro/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2504.07981) |
| **Tags** | `Agent`, `Grounding`, `MultiModal` |
| **Metrics** | `accuracy`, `text_acc`, `icon_acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,581 |
| Prompt Length (Mean) | 319.22 chars |
| Prompt Length (Min/Max) | 295 / 395 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `CAD` | 261 | 313.18 | 296 | 344 |
| `Creative` | 341 | 318.02 | 296 | 395 |
| `Dev` | 299 | 329.79 | 296 | 392 |
| `OS` | 196 | 317.57 | 297 | 382 |
| `Office` | 230 | 320.76 | 296 | 372 |
| `Scientific` | 254 | 314.44 | 295 | 353 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,581 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 1920x1080 - 6016x3384 |
| Formats | png |


## Sample Example

**Subset**: `CAD`

```json
{
  "input": [
    {
      "id": "2d87e94f",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~933.3KB]"
        },
        {
          "text": "Identify the UI element for the instruction and give a single click point. Coordinates must be normalized to the range 0 to 1 relative to the image size. Do not output a bounding box.\nInstruction: Mark dimensions\nEnd your reply with the final answer on its own last line, formatted exactly as: Answer: [x, y]"
        }
      ]
    }
  ],
  "target": "[0.1672, 0.0435, 0.1802, 0.1019]",
  "id": 0,
  "group_id": 0,
  "subset_key": "CAD",
  "metadata": {
    "id": "inventor_windows_0",
    "sent_size": [
      3840,
      1080
    ],
    "bbox_norm": [
      0.1671875,
      0.04351851851851852,
      0.18020833333333333,
      0.10185185185185185
    ],
    "ui_type": "text",
    "application": "inventor",
    "platform": "windows"
  }
}
```

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets screenspot_pro \
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
    datasets=['screenspot_pro'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
