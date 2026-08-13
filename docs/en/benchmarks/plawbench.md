# PLawBench


## Overview

PLawBench is a rubric-based benchmark that evaluates large language models on real-world Chinese legal practice.
It mirrors the workflow of a practising lawyer across three hierarchical levels: eliciting facts during a public
legal consultation, analysing a case with structured legal reasoning, and drafting professional legal documents.
Every item ships with a rubric annotated by legal experts, and grading is performed by an LLM judge against that
rubric rather than against a single reference answer.

## Task Description

- **Task Type**: Open-ended Chinese legal generation graded with expert rubrics
- **Input**: A client statement, or a case description plus a legal question
- **Output**: A question list, a structured case analysis, or a full legal document
- **Domain**: Chinese legal practice (personal affairs, marriage and family, corporate governance, intellectual
  property, criminal and civil litigation, cross-border matters, labour, environmental safety, and more)

## Key Features

- 280 samples split into four subsets, one per PLawBench task:
  - `case_analysis` (250): case analysis scored on four dimensions — conclusion, case facts, reasoning, and cited
    statutes. Answers must follow the 【结论】/【案件事实】/【推理过程】/【法条依据】 structure.
  - `legal_consultation` (18): the model plays a lawyer and must produce 10-25 verifiable follow-up questions that
    surface the facts the client omitted or distorted.
  - `plaintiff_statement` (6): drafting a statement of complaint from the client's account.
  - `defendant_statement` (6): drafting a statement of defense from the client's account and the opposing complaint.
- Client statements are deliberately vague, emotional, or misleading, so models must detect traps instead of
  restating the client's claims.
- Task prompts and judge prompts are ported verbatim from the official release, and the `case_analysis` rubric
  retains its per-dimension point allocation.

## Evaluation Notes

- Requires an LLM judge: run with `judge_strategy='llm'` (or `'auto'`, which enables the judge for this benchmark)
  and provide `judge_model_args`. `judge_strategy='rule'` is not supported.
- Metrics are point ratios in `[0, 1]`. `acc` is reported for every subset; `case_analysis` additionally reports
  `conclusion_acc`, `fact_acc`, `reasoning_acc`, and `law_acc`. These map one-to-one onto the official leaderboard
  columns: `legal_consultation` is Task1, `case_analysis` is Task2-Avg with its four dimensions, and the two
  drafting subsets are Task3-Plaintiff and Task3-Defendant.
- Compare per-subset scores, not the `OVERALL` row. `OVERALL` is a per-sample mean, so `case_analysis` dominates it
  (250 of 280 samples). The paper's `Overall` column is an equal-weighted mean of the three task scores, which
  matches its published table far more closely (mean absolute error 0.72 versus 2.87 for a sample-weighted mean,
  fitted across the 24 models in the official ranking).
- Rubric point totals come from the dataset, not from the judge output, and awarded points are clamped into
  `[0, max_points]`, so a judge that mis-reports the denominator cannot distort the score.
- The judge output template for `case_analysis` is repaired relative to the official script, which ships malformed
  JSON and pins the conclusion section to zero points; every section is graded on its rubric allocation here.
- Judge requests are retried up to `judge_retries` times when the response cannot be parsed; a sample that still
  fails is scored 0 and flagged via `judge_failed` in the review metadata.
- Case-analysis judging returns a long per-item breakdown. Give the judge a generous `max_tokens`
  (for example 8192) in `judge_model_args.generation_config`.
- The drafting subsets ask for a 2,500-3,000 character legal document, so the evaluated model also needs a generous
  `generation_config.max_tokens`. A truncated filing is graded as an incomplete document and scores near zero, which
  depresses Task3 for reasons unrelated to legal ability.

Resources: [GitHub](https://github.com/skylenage/PLawbench) |
[Dataset](https://modelscope.cn/datasets/evalscope/PLawBench)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `plawbench` |
| **Dataset ID** | [evalscope/PLawBench](https://modelscope.cn/datasets/evalscope/PLawBench/summary) |
| **Paper** | [Paper](https://github.com/skylenage/PLawbench) |
| **Tags** | `Chinese`, `Knowledge`, `QA`, `Reasoning` |
| **Metrics** | `accuracy`, `conclusion_acc`, `fact_acc`, `reasoning_acc`, `law_acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 280 |
| Prompt Length (Mean) | 2669.88 chars |
| Prompt Length (Min/Max) | 1267 / 5890 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `case_analysis` | 250 | 2720.18 | 2137 | 4873 |
| `legal_consultation` | 18 | 1851.94 | 1512 | 2611 |
| `plaintiff_statement` | 6 | 1494.67 | 1267 | 2050 |
| `defendant_statement` | 6 | 4203 | 2794 | 5890 |

## Sample Example

**Subset**: `case_analysis`

```json
{
  "input": [
    {
      "id": "06c8a1d7",
      "content": "\n## 角色\n\n你是一名具有十年以上执业经验的法律实务专家，精通中国现行法律法规与司法实践。你擅长将复杂的法律问题分解为清晰的逻辑模块，并严格依据“结论先行、事实为重、推理严密、依据支撑”的专业风格进行解答。\n\n## 核心要求\n\n1. 严格顺序：回答必须按照以下四部分顺序展开，并使用对应标题：\n【结论】\n【案件事实】\n【推理过程】\n【法条依据】\n2. 内容规范：\n结论：直接、明确，针对提问的核心争议点给出肯定或否定的判断。\n案件事实：基于用户提供的案情，简明、客观地摘录与法律判断 ... [TRUNCATED 1912 chars] ... 并赔偿精神损害抚慰金。庭审中查明，某摄影服务公司已完成除摄像外的其他服务项目；某文化传媒公司系独立法人，其工作人员在操作设备时存在重大过失。另查，某甲在签订合同时未特别声明婚礼录像的重要性，但合同附件中列有\"全程跟拍记录\"服务项目。某摄影服务公司辩称其仅需承担合同违约责任，精神损害赔偿缺乏依据。某文化传媒公司以非合同相对方为由拒绝承担责任。\n\n## 问题\n以【结论 + 案情简述 + 分析过程+依据法条】的逻辑回答以下问题：在上述案例中，某甲能否向某摄影服务公司主张精神损害赔偿？\n"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "case_analysis",
  "metadata": {
    "id": "case_analysis-1",
    "task": "case_analysis",
    "judge_type": "case_analysis",
    "category": "个人生活",
    "rubrics": "[{\"criterion\": \"【结论得分】\\n(+5分) 某甲有权向某摄影服务公司主张精神损害赔偿。\", \"points\": \"5\", \"tags\": \"结论得分\"}, {\"criterion\": \"【案情简述得分】\\n(+5分) 某甲与某摄影服务公司签订《婚庆服务合同》，并支付全款，合同附件明确包含\\\"全程跟拍记录\\\"服务项目。\\n(+5分) 婚礼当日，某摄影服务公司未经告知将摄像服务转包给文化传媒公司。\\n(+5分) 文化传媒公司工作室将录像全部丢失，未能交付原告。\\n(+ ... [TRUNCATED 762 chars] ... 人具有人身意义的特定物造成严重精神损害的，被侵权人有权请求精神损害赔偿。\\n（+5分）《最高人民法院关于确定民事侵权精神损害赔偿责任若干问题的解释》第五条\\n精神损害的赔偿数额根据以下因素确定：(一)侵权人的过错程度，但是法律另有规定的除外；(二)侵权行为的目的、方式、场合等具体情节；(三)侵权行为所造成的后果；(四)侵权人的获利情况；(五)侵权人承担责任的经济能力；(六)受理诉讼法院所在地的平均生活水平。\", \"points\": \"15\", \"tags\": \"法条依据得分\"}]",
    "max_points": 60,
    "prompt": "\n## 角色\n\n你是一名具有十年以上执业经验的法律实务专家，精通中国现行法律法规与司法实践。你擅长将复杂的法律问题分解为清晰的逻辑模块，并严格依据“结论先行、事实为重、推理严密、依据支撑”的专业风格进行解答。\n\n## 核心要求\n\n1. 严格顺序：回答必须按照以下四部分顺序展开，并使用对应标题：\n【结论】\n【案件事实】\n【推理过程】\n【法条依据】\n2. 内容规范：\n结论：直接、明确，针对提问的核心争议点给出肯定或否定的判断。\n案件事实：基于用户提供的案情，简明、客观地摘录与法律判断 ... [TRUNCATED 1912 chars] ... 并赔偿精神损害抚慰金。庭审中查明，某摄影服务公司已完成除摄像外的其他服务项目；某文化传媒公司系独立法人，其工作人员在操作设备时存在重大过失。另查，某甲在签订合同时未特别声明婚礼录像的重要性，但合同附件中列有\"全程跟拍记录\"服务项目。某摄影服务公司辩称其仅需承担合同违约责任，精神损害赔偿缺乏依据。某文化传媒公司以非合同相对方为由拒绝承担责任。\n\n## 问题\n以【结论 + 案情简述 + 分析过程+依据法条】的逻辑回答以下问题：在上述案例中，某甲能否向某摄影服务公司主张精神损害赔偿？\n"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `judge_retries` | `int` | `3` | Maximum attempts per rubric judge request before the sample is scored as 0. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets plawbench \
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
    datasets=['plawbench'],
    dataset_args={
        'plawbench': {
            # subset_list: ['case_analysis', 'legal_consultation', 'plaintiff_statement']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
