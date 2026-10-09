# Evaluating System One Choice Models

Use `eval_type='systemone_api'` for text decision models that return a selected option. EvalScope converts raw questions, context, demonstrations and complete candidate lists into System One requests, then computes accuracy from the returned labels.

## Bailian example

```python
import os

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='decision-model-preview',
    eval_type='systemone_api',
    api_url='https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ['DASHSCOPE_API_KEY'],
    datasets=['ceval', 'banking77'],
    dataset_args={
        'ceval': {
            'subset_list': ['computer_network'],
            'few_shot_num': 5,
            'system_prompt': 'Choose the best answer using the question and examples.',
        },
    },
    generation_config={'timeout': 60, 'retries': 3, 'retry_interval': 2},
    eval_batch_size=8,
    limit=5,
))
```

The API base URL must end in `/v1`; requests append `/systemone`. Bailian also supports workspace domains; consult its current [API documentation](https://help.aliyun.com/zh/model-studio/decision-model-api). TypeSafe Jev uses the same evaluation type with its own API base URL, key and pinned model version.

## TypeSafe Jev example

Set `TYPESAFE_API_KEY` in the environment, then run:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='jev-1.13.0',
    eval_type='systemone_api',
    api_url='https://api.typesafe.ai/v1',
    datasets=['boolq'],
    limit=2,
    eval_batch_size=1,
))
```

An explicit `api_key` takes precedence. When it is omitted, `None`, empty or the default placeholder `EMPTY`, the adapter reads `TYPESAFE_API_KEY` only for the official HTTPS host `api.typesafe.ai`. Bailian and custom endpoints require an explicit key; the TypeSafe environment key is not forwarded to them. See the [TypeSafe API reference](https://docs.typesafe.ai/api).

## Instructions and demonstrations

- `system_prompt` becomes part of the task instructions. System One has no native system message role or chat-role priority.
- `few_shot_num` retains the benchmark's defaults, sampling rules and permitted counts. Demonstrations go into `state.examples`; GPQA and SuperGPQA retain their fixed examples.
- `dataset_args.<benchmark>.choice_instructions` replaces the default decision instructions. Refer to state fields such as `question` and `examples` using backticks.
- Default generation formatting and reasoning instructions are removed. Explicit generation templates, output filters and `use_cot` are incompatible.
- Configure transport controls such as timeout, retries and concurrency. Temperature, max_tokens, streaming and generation-specific parameters are unsupported.

## Supported benchmarks

`mmlu`, `ceval`, `cmmlu`, `mmlu_pro`, `arc`, `gpqa_diamond`, `hellaswag`, `winogrande`, `general_mcq`, `musr`, `super_gpqa`, `anli`, `boolq`, `banking77`, `contract_nli`, `reward_bench`.

- ARC-C is the `ARC-Challenge` subset of `arc`.
- ANLI exposes `r1`, `r2`, `r3`: test evaluation with demonstrations from the corresponding training round.
- BoolQ uses original validation labels and a binary Choice request.
- BANKING77 retains all 77 intents without candidate pruning.
- ContractNLI uses full-document `contractnli_b`, evaluating classification only, without evidence extraction.
- RewardBench uses v1 filtered pairs with deterministic position shuffling. Overall accuracy is weighted by sample count, not the official category-weighted leaderboard score.

The five new benchmarks default to ModelScope. ContractNLI uses the `contractnli_b` configuration of `evalscope/contract-nli`. Their `local_path` and `dataset_revision` settings select local data or pin a dataset version. Images, audio, video, multiple-correct questions, tools and LLM judging are unsupported.

## Results and reproducibility

Prediction JSONL files retain labels, probabilities and provider confidence in `model_output.choice_result`. The complete request, response and request hash are stored in `model_output.metadata`. Two-decimal probability distributions are validated with their rounding tolerance and stored without renormalization.

Configurations and reports record the Choice protocol and actual shot count. Execution statistics distinguish successes, failures and incomplete runs. Invalid responses terminate a run by default; with `ignore_errors=True`, failed samples are excluded and their coverage is reported.

Pin model and dataset versions, splits, demonstrations and instructions for comparisons. Prediction caches require matching requests; `rerun_review=True` refuses changed requests. A model alias cannot guarantee an unchanged remote model version.

References: [TypeSafe API](https://docs.typesafe.ai/api), [public Jev evaluation code](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking). Scores obtained with different splits, shots, reasoning settings or aggregation metrics are not directly comparable.
