# CrossSum-IN


## Overview

CrossSum-IN evaluates summarization of English BBC news in English and 29 languages from India and nearby regions.

## Task Description

- **Task Type**: Cross-lingual Summarization
- **Input**: A news article in English
- **Output**: A one-sentence summary in the subset's language
- **Domain**: BBC news

## Key Features

- 12,463 test articles across 30 summary-language subsets.
- Uses the paper's prompt and one example from the same language's `train` split by default.

## Evaluation Notes

- Scores the first non-empty line using corpus SacreBLEU chrF (character order 6, word order 0, beta 2).
- The `indic` category's `macro_score` gives equal weight to each language. Its `score` weights by sample count.
- Articles are untruncated. The longest examples require a long context window.
- Licensed for non-commercial use under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `crosssum_in` |
| **Dataset ID** | [google/IndicGenBench_crosssum_in](https://modelscope.cn/datasets/google/IndicGenBench_crosssum_in/summary) |
| **Paper** | [Paper](https://aclanthology.org/2024.acl-long.595/) |
| **Tags** | `MultiLingual` |
| **Metrics** | `chrf` |
| **Default Shots** | 1-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 12,463 |
| Prompt Length (Mean) | 8303.71 chars |
| Prompt Length (Min/Max) | 1811 / 80198 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `en` | 500 | 8916.97 | 5066 | 76155 |
| `as` | 500 | 8917.97 | 5067 | 76156 |
| `awa` | 500 | 5879.97 | 2029 | 73118 |
| `bgc` | 500 | 8224.97 | 4374 | 75463 |
| `bho` | 500 | 8043.97 | 4193 | 75282 |
| `bn` | 155 | 6583.78 | 3341 | 21182 |
| `bo` | 499 | 6299.98 | 2445 | 73534 |
| `brx` | 500 | 5661.97 | 1811 | 72900 |
| `gbm` | 500 | 9939.97 | 6089 | 77178 |
| `gom` | 500 | 6356.97 | 2506 | 73595 |
| `gu` | 104 | 11224.04 | 7178 | 77591 |
| `hi` | 481 | 7252.13 | 3758 | 74847 |
| `hne` | 499 | 7089.83 | 3237 | 74326 |
| `hoj` | 500 | 9224.97 | 5374 | 76463 |
| `kn` | 500 | 8913.97 | 5063 | 76152 |
| `mai` | 499 | 9254.26 | 5404 | 76493 |
| `ml` | 500 | 8941.97 | 5091 | 76180 |
| `mni` | 497 | 6706.51 | 2855 | 73944 |
| `mr` | 118 | 9623.2 | 5447 | 76164 |
| `mup` | 500 | 10125.97 | 6275 | 77364 |
| `mwr` | 500 | 10087.97 | 6237 | 77326 |
| `ne` | 163 | 8686.5 | 5510 | 17316 |
| `or` | 500 | 8932.97 | 5082 | 76171 |
| `pa` | 136 | 10065.45 | 5956 | 27075 |
| `ps` | 154 | 8983.71 | 5350 | 76439 |
| `sa` | 499 | 6341.02 | 2488 | 73577 |
| `sat` | 500 | 9233.97 | 5383 | 76472 |
| `ta` | 315 | 6187.91 | 2738 | 73435 |
| `te` | 212 | 13614.55 | 9163 | 80198 |
| `ur` | 632 | 9770.29 | 6269 | 77358 |

## Sample Example

**Subset**: `en`

```json
{
  "input": [
    {
      "id": "7d9c3fa5",
      "content": "I will first show a news article in English and then provide a summary of it in the English language.\n\nSummarize the following article: Dave LeeNorth America technology reporter Twitter said it removed 936 accounts it said were being used to  ... [TRUNCATED 11284 chars] ... y expensive. That's why, says Gagandeep Kang, a good vaccine for India should cost below 50 cents a shot, be plentifully available and delivered as a single dose. What do I need to know about the coronavirus? Follow Soutik on Twitter\nSummary:"
    }
  ],
  "target": "When it comes to vaccine making, India is a powerhouse.",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "article": "Soutik BiswasIndia correspondent It runs a massive immunisation programme, makes 60% of the world's vaccines and is home to half a dozen major manufacturers, including Serum Institute of India - the largest in the world. Not surprisingly, the ... [TRUNCATED 6503 chars] ... ld be very expensive. That's why, says Gagandeep Kang, a good vaccine for India should cost below 50 cents a shot, be plentifully available and delivered as a single dose. What do I need to know about the coronavirus? Follow Soutik on Twitter",
    "language": "en",
    "language_name": "English",
    "source_url": "https://www.bbc.com/news/world-asia-india-55048925",
    "target_url": "https://www.bbc.com/news/world-asia-india-55048925"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
I will first show a news article in English and then provide a summary of it in the {language} language.

Summarize the following article: {article}
Summary:
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets crosssum_in \
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
    datasets=['crosssum_in'],
    dataset_args={
        'crosssum_in': {
            # subset_list: ['en', 'as', 'awa']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
