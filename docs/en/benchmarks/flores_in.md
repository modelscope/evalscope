# FLORES-IN


## Overview

FLORES-IN evaluates translation between English and 29 languages from India and nearby regions.

## Task Description

- **Task Type**: Machine Translation
- **Input**: A sentence in English (`en-xx` subsets) or in the paired language (`xx-en` subsets)
- **Output**: Its translation into the other language
- **Domain**: General-domain translation between English and 29 languages from India and nearby regions

## Key Features

- 58,696 test sentences across 58 subsets, one for each language and direction.
- Uses the paper's prompt and one example from the same subset's `dev` split by default.

## Evaluation Notes

- Scores the first non-empty line using corpus SacreBLEU chrF (character order 6, word order 0, beta 2).
- Each direction's `macro_score` gives equal weight to its 29 languages.
- Licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `flores_in` |
| **Dataset ID** | [google/IndicGenBench_flores_in](https://modelscope.cn/datasets/google/IndicGenBench_flores_in/summary) |
| **Paper** | [Paper](https://aclanthology.org/2024.acl-long.595/) |
| **Tags** | `MachineTranslation`, `MultiLingual` |
| **Metrics** | `chrf` |
| **Default Shots** | 1-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `dev` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 58,696 |
| Prompt Length (Mean) | 461.71 chars |
| Prompt Length (Min/Max) | 234 / 946 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `en-as` | 1,012 | 581.4 | 486 | 819 |
| `en-awa` | 1,012 | 554.4 | 459 | 792 |
| `en-bgc` | 1,012 | 425.4 | 330 | 663 |
| `en-bho` | 1,012 | 566.4 | 471 | 804 |
| `en-bn` | 1,012 | 389.4 | 294 | 627 |
| `en-bo` | 1,012 | 491.4 | 396 | 729 |
| `en-brx` | 1,012 | 436.4 | 341 | 674 |
| `en-gbm` | 1,012 | 441.4 | 346 | 679 |
| `en-gom` | 1,012 | 441.4 | 346 | 679 |
| `en-gu` | 1,012 | 493.4 | 398 | 731 |
| `en-hi` | 1,012 | 401.4 | 306 | 639 |
| `en-hne` | 1,012 | 554.4 | 459 | 792 |
| `en-hoj` | 1,012 | 425.4 | 330 | 663 |
| `en-kn` | 1,012 | 507.4 | 412 | 745 |
| `en-mai` | 1,012 | 418.4 | 323 | 656 |
| `en-ml` | 1,012 | 454.4 | 359 | 692 |
| `en-mni` | 1,012 | 540.4 | 445 | 778 |
| `en-mr` | 1,012 | 398.4 | 303 | 636 |
| `en-mup` | 1,012 | 422.4 | 327 | 660 |
| `en-mwr` | 1,012 | 421.4 | 326 | 659 |
| `en-ne` | 1,012 | 422.4 | 327 | 660 |
| `en-or` | 1,012 | 386.4 | 291 | 624 |
| `en-pa` | 1,012 | 448.4 | 353 | 686 |
| `en-ps` | 1,012 | 496.4 | 401 | 734 |
| `en-sa` | 1,012 | 423.4 | 328 | 661 |
| `en-sat` | 1,012 | 400.4 | 305 | 638 |
| `en-ta` | 1,012 | 558.4 | 463 | 796 |
| `en-te` | 1,012 | 432.4 | 337 | 670 |
| `en-ur` | 1,012 | 459.4 | 364 | 697 |
| `as-en` | 1,012 | 488.44 | 399 | 693 |
| `awa-en` | 1,012 | 635.65 | 543 | 887 |
| `bgc-en` | 1,012 | 417.0 | 331 | 625 |
| `bho-en` | 1,012 | 325.09 | 234 | 565 |
| `bn-en` | 1,012 | 461.51 | 372 | 667 |
| `bo-en` | 1,012 | 661.39 | 557 | 946 |
| `brx-en` | 1,012 | 455.85 | 348 | 708 |
| `gbm-en` | 1,012 | 436.53 | 347 | 642 |
| `gom-en` | 1,012 | 436.71 | 348 | 646 |
| `gu-en` | 1,012 | 483.93 | 397 | 708 |
| `hi-en` | 1,012 | 424.5 | 336 | 676 |
| `hne-en` | 1,012 | 465.43 | 382 | 666 |
| `hoj-en` | 1,012 | 417.48 | 327 | 627 |
| `kn-en` | 1,012 | 450.4 | 357 | 702 |
| `mai-en` | 1,012 | 555.87 | 464 | 778 |
| `ml-en` | 1,012 | 387.57 | 288 | 616 |
| `mni-en` | 1,012 | 424.16 | 327 | 642 |
| `mr-en` | 1,012 | 435.47 | 338 | 659 |
| `mup-en` | 1,012 | 424.53 | 336 | 636 |
| `mwr-en` | 1,012 | 421.22 | 326 | 636 |
| `ne-en` | 1,012 | 431.4 | 340 | 668 |
| `or-en` | 1,012 | 535.24 | 440 | 756 |
| `pa-en` | 1,012 | 490.84 | 396 | 739 |
| `ps-en` | 1,012 | 461.0 | 375 | 665 |
| `sa-en` | 1,012 | 561.74 | 468 | 793 |
| `sat-en` | 1,012 | 465.13 | 370 | 693 |
| `ta-en` | 1,012 | 475.31 | 371 | 727 |
| `te-en` | 1,012 | 348.9 | 257 | 577 |
| `ur-en` | 1,012 | 408.38 | 314 | 628 |

## Sample Example

**Subset**: `en-as`

```json
{
  "input": [
    {
      "id": "7c97bce9",
      "content": "Translate the following:\n\nTo Assamese: In the 1480s, when its construction was begun by Lord Hastings, the country was relatively peaceful and defense was only required against small bands of roving marauders.\nOutput: 1480 চনত লর্ড হেষ্টিংছে  ... [TRUNCATED 105 chars] ... াওডাৰৰ সৰু সৰু গোটৰ বিৰুদ্ধেহে সেনেবাহিনীৰ দৰকাৰ হৈছিল।\n\nTranslate the following:\n\nTo Assamese: Subcultures bring together like-minded individuals who feel neglected by societal standards and allow them to develop a sense of identity.\nOutput:"
    }
  ],
  "target": "উপসংস্কৃতিসমূহে ব্যক্তি বিশেষৰ একেলগে থকা মনোভাৱ আনি দিয়ে যিসকলে সমাজৰ পৰা অৱহেলাৰ সন্মুখীন হৈছিল আৰু তেওঁলোকক নিজা পৰিচয়ৰ মনোভাৱ বিকশিত কৰাত সহায় কৰে।",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "source": "Subcultures bring together like-minded individuals who feel neglected by societal standards and allow them to develop a sense of identity.",
    "source_language": "en",
    "target_language": "as",
    "target_language_name": "Assamese"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Translate the following:

To {language}: {source}
Output:
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets flores_in \
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
    datasets=['flores_in'],
    dataset_args={
        'flores_in': {
            # subset_list: ['en-as', 'en-awa', 'en-bgc']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
