# ArxivRollBench-Full

## Overview

ArxivRollBench is a rolling benchmark built from recent arXiv papers. It evaluates whether large language models can reason over fresh scientific text through three task formats: sequencing, cloze, and next-fragment prediction.

## Task Description

- **Task Type**: Multiple-choice scientific text reasoning
- **Input**: Recent arXiv text fragments with four answer choices
- **Output**: Single correct answer letter (A, B, C, or D)
- **Domains**: Computer Science, Quantitative Finance, Mathematics, Physics, Statistics, Quantitative Biology, Economics, and Electrical Engineering/System Science
- **Releases**: 2024b, 2025a, and 2026a rolling snapshots

## Key Features

- Time-aware benchmark snapshots reduce contamination-related overestimation
- Covers multiple arXiv domains and scientific writing styles
- Includes sequencing, cloze, and prediction formats under the SCP framework
- Compact `-50` split is suitable for cost-controlled API evaluation
- Full split is available as `arxivrollbench_full`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- The default `arxivrollbench` benchmark uses compact `-50` datasets
- Use `arxivrollbench_full` for the complete public splits
- Each subset is loaded from the public ModelScope mirror under the `liangzid` namespace
- Answers are normalized to A-D and evaluated with accuracy

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `arxivrollbench_full` |
| **Dataset ID** | [liangzid/arxivrollbench-full](https://modelscope.cn/datasets/liangzid/arxivrollbench-full/summary) |
| **Paper** | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/41098) |
| **Tags** | `Knowledge`, `MCQ`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 245,433 |
| Prompt Length (Mean) | 1499.93 chars |
| Prompt Length (Min/Max) | 307 / 28864 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `2024b_cs_s` | 2,931 | 962.16 | 574 | 4774 |
| `2024b_cs_c` | 2,377 | 307 | 307 | 307 |
| `2024b_cs_p` | 3,166 | 2663.27 | 793 | 10327 |
| `2024b_q_fin_s` | 852 | 1026.01 | 574 | 3549 |
| `2024b_q_fin_c` | 747 | 307 | 307 | 307 |
| `2024b_q_fin_p` | 881 | 3207.96 | 793 | 16189 |
| `2024b_math_s` | 2,107 | 886.2 | 574 | 3466 |
| `2024b_math_c` | 1,238 | 307 | 307 | 307 |
| `2024b_math_p` | 2,532 | 2295.3 | 793 | 11911 |
| `2024b_physics_s` | 1,966 | 984.28 | 575 | 4225 |
| `2024b_physics_c` | 1,482 | 307 | 307 | 307 |
| `2024b_physics_p` | 2,141 | 3166.87 | 793 | 28864 |
| `2024b_stat_s` | 3,482 | 985.03 | 574 | 6098 |
| `2024b_stat_c` | 2,800 | 307 | 307 | 307 |
| `2024b_stat_p` | 3,704 | 3000.94 | 793 | 15321 |
| `2024b_q_bio_s` | 1,485 | 1039.14 | 574 | 3895 |
| `2024b_q_bio_c` | 1,318 | 307 | 307 | 307 |
| `2024b_q_bio_p` | 1,550 | 3332.41 | 804 | 16126 |
| `2024b_econ_s` | 879 | 1023.84 | 576 | 3421 |
| `2024b_econ_c` | 764 | 307 | 307 | 307 |
| `2024b_econ_p` | 919 | 3176.67 | 851 | 15040 |
| `2024b_eess_s` | 3,771 | 1014.36 | 574 | 4356 |
| `2024b_eess_c` | 3,278 | 307 | 307 | 307 |
| `2024b_eess_p` | 3,976 | 3048.85 | 793 | 17290 |
| `2025a_cs_s` | 12,806 | 981.57 | 574 | 5696 |
| `2025a_cs_c` | 11,244 | 307 | 307 | 307 |
| `2025a_cs_p` | 13,331 | 2823.48 | 793 | 20389 |
| `2025a_q_fin_s` | 851 | 1013.21 | 576 | 2609 |
| `2025a_q_fin_c` | 758 | 307 | 307 | 307 |
| `2025a_q_fin_p` | 884 | 3128.37 | 793 | 13025 |
| `2025a_math_s` | 10,362 | 908.79 | 574 | 6001 |
| `2025a_math_c` | 6,344 | 307 | 307 | 307 |
| `2025a_math_p` | 12,145 | 2444.85 | 793 | 12037 |
| `2025a_physics_s` | 10,696 | 1002.06 | 574 | 4761 |
| `2025a_physics_c` | 8,358 | 307 | 307 | 307 |
| `2025a_physics_p` | 11,595 | 3369.68 | 793 | 25245 |
| `2025a_stat_s` | 5,288 | 985.58 | 574 | 8627 |
| `2025a_stat_c` | 4,285 | 307 | 307 | 307 |
| `2025a_stat_p` | 5,589 | 2935.37 | 793 | 15676 |
| `2025a_q_bio_s` | 1,598 | 1043.55 | 574 | 3115 |
| `2025a_q_bio_c` | 1,443 | 307 | 307 | 307 |
| `2025a_q_bio_p` | 1,669 | 3370.82 | 796 | 18074 |
| `2025a_econ_s` | 951 | 998.31 | 574 | 2900 |
| `2025a_econ_c` | 827 | 307 | 307 | 307 |
| `2025a_econ_p` | 982 | 3176.93 | 793 | 11038 |
| `2025a_eess_s` | 8,171 | 1011.86 | 574 | 3844 |
| `2025a_eess_c` | 7,155 | 307 | 307 | 307 |
| `2025a_eess_p` | 8,577 | 3042.87 | 793 | 18934 |
| `2026a_cs_s` | 1,857 | 981.82 | 574 | 3532 |
| `2026a_cs_c` | 1,648 | 307 | 307 | 307 |
| `2026a_cs_p` | 1,933 | 2724.96 | 814 | 11328 |
| `2026a_q_fin_s` | 986 | 985.79 | 574 | 2961 |
| `2026a_q_fin_c` | 886 | 307 | 307 | 307 |
| `2026a_q_fin_p` | 1,046 | 2727.72 | 802 | 10072 |
| `2026a_math_s` | 2,435 | 869.86 | 574 | 3795 |
| `2026a_math_c` | 1,600 | 307 | 307 | 307 |
| `2026a_math_p` | 2,777 | 1953.57 | 808 | 12053 |
| `2026a_physics_s` | 1,863 | 1007.76 | 574 | 3813 |
| `2026a_physics_c` | 1,575 | 307 | 307 | 307 |
| `2026a_physics_p` | 2,019 | 3072.96 | 798 | 13540 |
| `2026a_stat_s` | 3,126 | 964.56 | 574 | 3136 |
| `2026a_stat_c` | 2,627 | 307 | 307 | 307 |
| `2026a_stat_p` | 3,322 | 2549.38 | 814 | 10028 |
| `2026a_q_bio_s` | 1,502 | 1020.61 | 574 | 3281 |
| `2026a_q_bio_c` | 1,373 | 307 | 307 | 307 |
| `2026a_q_bio_p` | 1,569 | 3074.52 | 806 | 11848 |
| `2026a_econ_s` | 914 | 995.97 | 574 | 3043 |
| `2026a_econ_c` | 828 | 307 | 307 | 307 |
| `2026a_econ_p` | 973 | 2858.55 | 818 | 11577 |
| `2026a_eess_s` | 4,200 | 1006.27 | 574 | 3698 |
| `2026a_eess_c` | 3,710 | 307 | 307 | 307 |
| `2026a_eess_p` | 4,409 | 2790.21 | 817 | 13794 |

## Sample Example

**Subset**: `2024b_cs_s`

```json
{
  "input": [
    {
      "id": "509c2daa",
      "content": "Answer the following ArxivRollBench multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nSelect the option that correctly compl ... [TRUNCATED 283 chars] ... rators can be used directly to verify representations of classical groups [12].\n**C**: In practice it is the generating set produced by the constructive recognition algorithms from [10, 11] as implemented in MAGMA\n\nA) CAB\nB) ACB\nC) BAC\nD) CAB"
    }
  ],
  "choices": [
    "CAB",
    "ACB",
    "BAC",
    "CAB"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "original_label": "Selection 2",
    "task_type": "s/c"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following ArxivRollBench multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets arxivrollbench_full \
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
    datasets=['arxivrollbench_full'],
    dataset_args={
        'arxivrollbench_full': {
            # subset_list: ['2024b_cs_s', '2024b_cs_c', '2024b_cs_p']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


