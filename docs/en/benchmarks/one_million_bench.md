# $OneMillion-Bench


## Overview

$OneMillion-Bench ($1M-Bench) evaluates how well language models and agents complete economically valuable,
expert-level professional work. The public release contains 400 bilingual tasks written and reviewed by domain
experts across finance, healthcare, industry, law, and natural science.

## Task Description

- **Task Type**: Open-ended professional question answering with rubric-based LLM judging
- **Input**: A realistic, context-heavy professional request in Chinese or English
- **Output**: A complete free-form professional analysis or deliverable
- **Domain**: Economics and finance, healthcare and medicine, industry, law, and natural sciences

## Key Features

- 400 zero-shot tasks, balanced across Chinese and global tracks and five professional domains (40 tasks per
  language-domain subset)
- Each task has 11-37 expert-authored criteria covering factual information, analytical reasoning, instruction
  following, and structure and formatting
- Every task includes both positive criteria and negative penalties, with rubric weights ranging from -20 to 12 in
  the hosted release
- Samples are exposed as ten language-domain subsets so both the paper's language tracks and domain breakdowns are
  visible in EvalScope reports

## Evaluation Notes

- An LLM judge is required. Configure `judge.strategy='llm'` (or `'auto'`) and `judge.models`; the official harness
  currently recommends Gemini 3.1 Pro Preview, but judge identity affects absolute scores and is not hard-coded here
- All rubrics for one response are judged together in one request using the official binary hit/miss instructions
- `expert_score` is the weighted sum of hit rubrics divided by the sum of positive weights, clipped to `[0, 1]`;
  `pass_rate` is 1 when `expert_score >= 0.7`, otherwise 0
- Judge replies must contain every rubric exactly once. Malformed replies and transport failures are excluded instead
  of being silently converted to zero scores
- The official study compares vanilla models, search-enabled models, and deep-research agents separately. This native
  adapter performs the benchmark's one-turn generation path; results from external tool-using agents are comparable
  only when their final responses are evaluated under the same judge configuration
- Tasks often require long, cited reports. Configure sufficiently large generation and judge `max_tokens` values;
  with one judge and one repeat, a full run performs 400 generation calls and 400 judge calls

Resources: [Paper](https://arxiv.org/abs/2603.07980) |
[GitHub](https://github.com/humanlaya/OneMillion-Bench) |
[Dataset](https://modelscope.cn/datasets/evalscope/OneMillion-Bench)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `one_million_bench` |
| **Dataset ID** | [evalscope/OneMillion-Bench](https://modelscope.cn/datasets/evalscope/OneMillion-Bench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2603.07980) |
| **Tags** | `Agent`, `Knowledge`, `MultiLingual`, `QA`, `Reasoning` |
| **Metrics** | `expert_score`, `pass_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 400 |
| Prompt Length (Mean) | 1470.49 chars |
| Prompt Length (Min/Max) | 105 / 15951 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `global_economics_and_finance` | 40 | 1810.97 | 868 | 3723 |
| `global_healthcare_and_medicine` | 40 | 2029.97 | 479 | 8870 |
| `global_industry` | 40 | 2581.15 | 453 | 9538 |
| `global_law` | 40 | 3277.78 | 404 | 15951 |
| `global_natural_sciences` | 40 | 1634.65 | 284 | 5775 |
| `cn_economics_and_finance` | 40 | 555.05 | 111 | 1169 |
| `cn_healthcare_and_medicine` | 40 | 584.38 | 135 | 2628 |
| `cn_industry` | 40 | 965.6 | 144 | 7755 |
| `cn_law` | 40 | 709.55 | 208 | 2041 |
| `cn_natural_sciences` | 40 | 555.83 | 105 | 1590 |

## Sample Example

**Subset**: `global_economics_and_finance`

```json
{
  "input": [
    {
      "id": "24273bb5",
      "content": "You are an international financial risk analyst. Based on the Financial Stability Report released by the Bank of England's Financial Policy Committee in October 2025, global financial markets may face a \"risk of sharp market correction\" if in ... [TRUNCATED 924 chars] ... ields, and global capital flows.\nSpecial Conditions: Use only information published before December 31, 2025; do not fabricate information; generated content must cite real URLs. The answer must be complete and useful; do not fake a response."
    }
  ],
  "target": "[{\"rubric_number\": 1, \"rubric_detail\": \"Mention the high weighting of top US companies (e.g., Top 5 or AI-related stocks) in the index (approximately 30% or more) and identify this as a source of systemic risk.\", \"rubric_weight\": 10, \"rubric_ ... [TRUNCATED 3986 chars] ... c_number\": 16, \"rubric_detail\": \"The report contains hollow concluding remarks (e.g., \\\"In summary,\\\" \\\"We look forward to\\\") or transitional sentences lacking substantive content.\", \"rubric_weight\": -2, \"rubric_tag\": \"Analytical Reasoning\"}]",
  "id": 0,
  "group_id": 0,
  "subset_key": "global_economics_and_finance",
  "metadata": {
    "id": "e1b94c86-b6c9-43f6-8251-2e513e5efc52",
    "case_id": 1663,
    "language": "global",
    "domain": "Economics and Finance",
    "topics": [
      "Economics and Finance",
      "Financing & M&A",
      "Mergers & Acquisitions"
    ],
    "time_sensitivity": {
      "time_sensitivity": "Weakly time-sensitive",
      "year_month": "2025-10",
      "day": "NA"
    },
    "question": "You are an international financial risk analyst. Based on the Financial Stability Report released by the Bank of England's Financial Policy Committee in October 2025, global financial markets may face a \"risk of sharp market correction\" if in ... [TRUNCATED 924 chars] ... ields, and global capital flows.\nSpecial Conditions: Use only information published before December 31, 2025; do not fabricate information; generated content must cite real URLs. The answer must be complete and useful; do not fake a response."
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets one_million_bench \
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
    datasets=['one_million_bench'],
    dataset_args={
        'one_million_bench': {
            # subset_list: ['global_economics_and_finance', 'global_healthcare_and_medicine', 'global_industry']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
