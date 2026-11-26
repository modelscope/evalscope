# CrossSum-IN


## Overview

CrossSum-IN is the cross-lingual summarization task of IndicGenBench (Google Research, ACL 2024). Built on
CrossSum, it pairs English BBC news articles with human-written summaries in 29 Indic languages, many of them
low-resource, and in English.

## Task Description

- **Task Type**: Cross-lingual Summarization
- **Input**: A news article in English
- **Output**: A one-sentence summary in the subset's language
- **Languages**: Assamese, Awadhi, Bengali, Bhojpuri, Bodo, Chhattisgarhi, Garhwali, Gujarati, Haryanvi, Hindi,
  Kannada, Konkani, Maithili, Malvi, Malayalam, Manipuri, Marathi, Marwari, Nepali, Odia, Pashto, Punjabi,
  Rajasthani, Sanskrit, Santali, Tamil, Telugu, Tibetan and Urdu, plus English

## Key Features

- 30 subsets named by summary language (`en`, `hi`, `sat`, ...), each evaluated on its released `test` file
  (104 to 632 articles per language, 12,463 in total)
- The `train` file of the same language (100 articles, 99 for `gbm`) provides the in-context examples
- Results are grouped into `indic` (the 29 Indic languages) and `english`, which the paper reports separately

## Evaluation Notes

- Default configuration uses **1-shot** evaluation, the paper's main setting. The example is the first `train`
  article of the same language, so every model sees the same prompt; set `few_shot_num=0` for zero-shot
- Prompt (Table 15 of the paper), with the English language names of its Table 7:
  `I will first show a news article in English and then provide a summary of it in the {language} language.\n\n`
  `Summarize the following article: {article}\nSummary:`. The example uses the same format followed by
  ` {summary}`, then a blank line before the test article
- Articles are not truncated, as in the paper: they average about 4,200 characters, 5% exceed 8,200, and 28
  (0.2%) exceed 32,768 characters (roughly 8k tokens), so a model with a short context window cannot take those
- The first non-empty line of the output is scored; use greedy decoding (`temperature=0`) as in the paper
- Metric: **chrF** from SacreBLEU with default settings (character order 6, word order 0, beta 2), computed at
  the corpus level per language. The Indic score is the macro-average of the 29 Indic languages, reported as the
  `macro_score` of the `indic` category (languages differ in size, so the category `score`, weighted by sample
  count, is a different number)
- Each sample also records its sentence-level chrF for inspection; only the corpus score is reported
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), from CrossSum: non-commercial
  use only


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
      "id": "4a179703",
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
