# System One Evaluation

System One is an API for decision models. The model receives content and decision instructions, then returns a choice, a yes/no probability or an ordinal score. EvalScope connects to these services through `eval_type='systemone_api'` and handles data loading, requests, scoring and reports.

This page covers connecting to Bailian or TypeSafe Jev, evaluating existing multiple-choice benchmarks and reading the results. For Choice, Noul and Score examples, custom data and Clef-Flash deployment, see the [decision evaluation best practice](../best_practice/decision_evaluation.md).

## Quick Start: Bailian

Install the evaluation and report browsing components, then set your Bailian API key:

```bash
pip install -U 'evalscope[service]'
export DASHSCOPE_API_KEY='<your-bailian-api-key>'
```

The following example uses `decision-model-preview` to evaluate five samples each from C-Eval's computer network subset and BANKING77:

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

`limit=5` caps the number of evaluation samples per subset. In contrast, `few_shot_num=5` supplies five demonstrations for each C-Eval question. This small run checks the evaluation workflow; it does not represent full-benchmark performance.

Set `api_url` to the service base URL ending in `/v1`. EvalScope appends `/systemone` automatically, so do not supply the full request path. The example uses the trial endpoint. If your account uses a workspace domain, replace the address and key according to the [Bailian API documentation](https://help.aliyun.com/zh/model-studio/decision-model-api).

## Use TypeSafe Jev

TypeSafe Jev uses the same evaluation type with a different model, endpoint and key. First set your TypeSafe API key:

```bash
export TYPESAFE_API_KEY='<your-typesafe-api-key>'
```

The following example evaluates two BoolQ samples:

```python
import os

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='jev-1.13.0',
    eval_type='systemone_api',
    api_url='https://api.typesafe.ai/v1',
    api_key=os.environ['TYPESAFE_API_KEY'],
    datasets=['boolq'],
    limit=2,
    eval_batch_size=1,
))
```

Choose a model version available to your account; see the [TypeSafe API reference](https://docs.typesafe.ai/api). Use a pinned version for comparisons rather than an alias whose underlying model may change.

When connecting to the official HTTPS host `api.typesafe.ai`, you can also omit `api_key` and let EvalScope read `TYPESAFE_API_KEY`. Bailian and custom endpoints require an explicit key.

## Configure Instructions and Requests

Existing multiple-choice questions are converted into a question, candidate answers and demonstrations before being sent to the model. Configure the task through `dataset_args.<benchmark>`:

| Parameter | Purpose |
| --- | --- |
| `system_prompt` | Adds task instructions; merged with the choice instructions without the priority of a chat system role |
| `few_shot_num` | Sets the demonstration count, following the benchmark's sampling rules and permitted counts; uses benchmark defaults when omitted |
| `choice_instructions` | Replaces the default choice instructions; use backticks to refer to input fields such as `question` and `examples` |

This mode does not ask the model to generate a natural-language answer. Formatting and reasoning instructions from the default generation template are not sent. Use `choice_instructions` to customize the task; do not set generation-oriented `prompt_template`, `few_shot_prompt_template`, `query_template` or output `filters`, or enable `use_cot`.

Configure request timeouts and retries in `generation_config`, and concurrency through `eval_batch_size`. In the example, `timeout=60` allows 60 seconds per attempt, `retries=3` allows three total attempts, and `retry_interval=2` waits two seconds between attempts. `temperature`, `max_tokens` and streaming generation are unsupported.

## Choose a Benchmark

The following existing benchmarks support text single-choice conversion:

| Task | Benchmark names |
| --- | --- |
| Knowledge and exams | `mmlu`, `ceval`, `cmmlu`, `mmlu_pro`, `arc`, `gpqa_diamond`, `super_gpqa` |
| Commonsense and reasoning | `hellaswag`, `winogrande`, `musr` |
| Reading comprehension and inference | `anli`, `boolq` |
| Text classification | `banking77`, `contract_nli` |
| Response preference | `reward_bench` |
| Custom multiple-choice data | `general_mcq` |

BoolQ also uses two-option Choice by default. To evaluate the probability of yes directly, set its `extra_params={'decision_kind': 'noul'}` within `dataset_args`. In the same `extra_params` dictionary, `decision_threshold` controls the yes/no cutoff and defaults to 0.5.

Native decision benchmarks preserve their question types and scoring rules and use zero-shot evaluation:

| Benchmark | Use case |
| --- | --- |
| [OpenDecisions](../benchmarks/open_decisions.md) | Evaluate Choice, Noul and Score for decision quality and probability performance |
| [JevBench public subset](../benchmarks/jevbench.md) | Evaluate public tasks using their rules; does not reproduce the full JevBench leaderboard |
| [STS-B Decision](../benchmarks/stsb.md) | Evaluate sentence similarity with Score while preserving continuous gold values |
| [General Decision](../advanced_guides/custom_dataset/decision.md) | Evaluate your own Choice, Noul or Score data |

Public datasets default to ModelScope. Use each benchmark's `local_path` for local data or `dataset_revision` to pin a dataset version. EvalScope's current System One integration supports text decisions only, without images, audio, video, multiple-correct questions, agents, tools or LLM judges.

```{dropdown} Benchmark-specific evaluation rules
- ARC-C is the `ARC-Challenge` subset of `arc`.
- ANLI exposes `r1`, `r2` and `r3`, using each round's test split for evaluation and its train split for demonstrations.
- BoolQ uses the original validation data.
- BANKING77 retains all 77 intents without candidate pruning.
- ContractNLI uses full contracts from the `contractnli_b` configuration of `evalscope/contract-nli`. It evaluates three-way classification, without evidence extraction.
- RewardBench uses v1 filtered data with deterministic response position shuffling. Overall accuracy is weighted by sample count, not by the official leaderboard's category weights.
- GPQA and SuperGPQA retain fixed demonstrations according to their benchmark rules.
```

## Read the Results

Results are saved under `outputs/<timestamp>/` by default: `reports/` contains aggregate reports, `predictions/` contains model outputs, and `reviews/` contains per-sample scores. Common metrics include:

| Metric | Interpretation |
| --- | --- |
| `acc` | Fraction of correct answers; higher is better |
| `nll`, `brier` | Probability prediction losses against the gold answer; lower is better |
| `ece` | Gap between predicted probabilities and observed event frequencies across probability bins; lower is better |
| `mae` | Mean absolute difference between probability-weighted Score values and gold scores; lower is better |

Available metrics depend on the question type and benchmark. For example, BANKING77 also reports `macro_f1`, while STS-B uses continuous-score MAE as its primary metric. See the [best practice](../best_practice/decision_evaluation.md) for OpenDecisions composite metrics.

Start the Web UI and open `http://127.0.0.1:9000`. Select a report on the evaluation page, then use the prediction browser to inspect predictions, gold answers and probability distributions:

```bash
evalscope service
```

Option probabilities describe the model's predicted distribution over answers, while `confidence` has a provider-defined meaning that may differ. The UI can switch between raw distributions and distributions used for scoring. Scoring may normalize a distribution to account for rounding; original response values are retained.

```{dropdown} Where requests and probabilities are saved
- `model_output.metadata.choice_request` in prediction JSONL stores the actual request sent to the model.
- `model_output.metadata.choice_response.answers.answer` stores the provider's answer, including the probabilities for that question type and optional provider confidence.
- `score.metadata.decision` in per-sample reviews stores the scoring distribution. `score.metadata.probabilities_normalized_for_scoring` records whether it was normalized.

To reproduce model input, use the saved request rather than only the question text displayed in the UI.
```

## Handle Failures and Reproduce Runs

Request failures and invalid responses stop evaluation by default. Set `ignore_errors=True` explicitly to skip failed samples and continue. Reports distinguish successful, failed and incomplete samples, so check coverage when comparing scores. Failures are usually excluded from ordinary metrics, but JevBench's `public_accuracy` counts missing predictions as wrong; consult each benchmark's scoring rules.

After fixing a problem, set `use_cache` to the original output directory to reuse completed results. Use `rerun_review` when only scoring needs to be repeated. See the [parameter reference](../get_started/parameters.md).

For comparisons, pin model and dataset versions, evaluation splits, demonstration counts and task instructions. Scores from different splits, demonstration counts or aggregation methods are not directly comparable with public leaderboards.
