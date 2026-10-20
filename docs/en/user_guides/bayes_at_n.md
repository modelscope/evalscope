# Bayes@N for repeated correctness scores

Set `aggregation='bayes_at_n'` to report the posterior mean and posterior standard deviation of mean correctness over a fixed question set. This binary version of [Bayes@N](https://arxiv.org/abs/2510.04265) uses an independent uniform Beta(1, 1) prior for each question. It supports `accuracy` and `exact_match` scores, including their `acc` and `em` aliases.

Choose one subset and use `repeats` for the number of independently sampled generations per question:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_MODEL',
    eval_type='openai_api',
    api_url='YOUR_API_URL',
    datasets=['general_mcq'],
    dataset_args={
        'general_mcq': {
            'local_path': 'custom_eval/text/mcq',
            'subset_list': ['example'],
            'aggregation': 'bayes_at_n',
        },
    },
    repeats=16,
    generation_config={'temperature': 0.7},
    judge={'strategy': 'rule'},
))
```

The normal CLI accepts the same configuration:

```bash
evalscope eval --model YOUR_MODEL --eval-type openai_api \
    --api-url YOUR_API_URL --datasets general_mcq --repeats 16 \
    --dataset-args '{"general_mcq":{"local_path":"custom_eval/text/mcq","subset_list":["example"],"aggregation":"bayes_at_n"}}' \
    --generation-config '{"temperature":0.7}'
```

The model's sampling settings should produce independent trials with the same prompt and grading rule. Repeating deterministic generations adds no independent evidence.

## Results

Each correctness metric produces two report identities with `dimensions.n` equal to `repeats`:

| Aggregation | Value | Report role |
| --- | --- | --- |
| `bayes_at_n` | Posterior mean of mean correctness | Quality score |
| `bayes_at_n_sd` | Posterior standard deviation of mean correctness | Diagnostic |

Both values use the 0–1 probability scale. The report displays the mean as a percentage and the SD as a plain number. `num` counts questions; metadata also records the number of generations, the prior, and full-precision posterior mean and SD. Normal report scalars retain EvalScope's four-decimal rounding. The SD describes uncertainty about this fixed question set's mean success probability. It does not measure variation from drawing a new benchmark question set.

For question `i` with `s_i` correct outcomes out of `N` generations, the posterior is `Beta(s_i + 1, N - s_i + 1)`. Its mean is `(s_i + 1)/(N + 2)` and its variance is `(s_i + 1)(N - s_i + 1)/((N + 2)^2(N + 3))`. Across `M` questions, Bayes@N averages the question means and reports `sqrt(sum(question variances))/M` as the SD. Outcomes remain grouped by question throughout this calculation.

For example, two questions with outcomes `[0, 0]` and `[1, 1]` give a posterior mean of `0.5` and posterior SD of `sqrt(3/160)`, about `0.1369`.

## Supported scope

Use an adapter that follows `DefaultDataAdapter`'s configured aggregation. Select exactly one subset. Multi-subset and dataset-collection runs are rejected because the generic report reducers cannot combine posterior SDs and the subset-local IDs do not establish disjoint question sets.

Every planned generation must have a usable binary correctness score. Missing grades, incomplete questions, duplicate generation positions, non-binary values and legacy reviews without generation positions raise an error. The adapter checks the scores against the loaded cohort, including questions with no completed generations. Cache resume can fill missing predictions; duplicate cache rows follow EvalScope's existing latest-row policy before aggregation.

Categorical rubrics, prior observations, partial-credit scores and benchmark-owned custom aggregators are outside this version's scope. The existing default aggregation is unchanged.
