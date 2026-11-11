# CLUEWSC


## Overview

CLUEWSC is the Chinese Winograd Schema Challenge task of [FewCLUE](https://github.com/CLUEbenchmark/FewCLUE). Each item asks whether a pronoun in a sentence refers to a given noun phrase, which requires commonsense reasoning rather than surface cues.

## Task Description

- **Task Type**: Binary coreference resolution as single-choice (Chinese)
- **Input**: A Chinese sentence, a pronoun and a candidate referent
- **Output**: A (是, refers to it) or B (否, does not)
- **Domain**: Chinese commonsense reasoning

## Key Features

- Official FewCLUE splits: `test_public` (976 items, balanced labels), `dev_few_all` (159) and `train_few_all` (160)
- Same question wording as the OpenCompass `FewCLUE_cluewsc_gen` configuration, so scores are comparable

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on `test_public`; set `eval_split='dev_few_all'` to score the dev split.
- Set `few_shot_num` to draw demonstrations from `train_few_all`.
- Answers should follow the format: "答案：[LETTER]"


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `cluewsc` |
| **Dataset ID** | [evalscope/cluewsc](https://modelscope.cn/datasets/evalscope/cluewsc/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2107.07498) |
| **Tags** | `Chinese`, `MCQ`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test_public` |
| **Train Split** | `train_few_all` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 976 |
| Prompt Length (Mean) | 182.99 chars |
| Prompt Length (Min/Max) | 135 / 271 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "e440caf5",
      "content": "回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式：\"答案：[LETTER]\"（不带引号），其中 [LETTER] 是 A,B 中的一个。\n\n问题：不过，在面子上，毛豆还下不来，一半是因为他确实很生气；另一半也是因为，他毛豆怎么能与他们做一路人。所以，他必须生气。有几次大王问他累不累，要不要喝水，后面的人立即送上矿泉水瓶子，他不理睬。\n此处，“他”是否指代“毛豆”？\n选项：\nA) 是\nB) 否\n"
    }
  ],
  "choices": [
    "是",
    "否"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "idx": 1180
  }
}
```

## Prompt Template

**Prompt Template:**
```text
回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。

问题：{question}
选项：
{choices}

```

<details>
<summary>Few-shot Template</summary>

```text
以下是一些示例问题：

{fewshot}

回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。

问题：{question}
选项：
{choices}

```

</details>

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cluewsc \
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
    datasets=['cluewsc'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
