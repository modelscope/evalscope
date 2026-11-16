# Evaluation diagnostics

Diagnostics connect a score to samples worth inspecting and available evidence. The default `facts` level reads existing reviews, predictions and Agent traces, computes deterministic statistics, and makes no LLM calls or evaluation requests.

## Usage

Analyze saved results:

```bash
evalscope analyze --run-dir outputs/20261116_120000 --level facts
```

A parent directory containing multiple runs also works. Repeat `--model <directory-model-ID>` or `--benchmark <name>` to restrict the scope. `--force` bypasses the cache. The command also refreshes each run's HTML report.

Enable analysis during evaluation:

```python
from evalscope import TaskConfig, run_task
from evalscope.api.analysis import AnalysisReportConfig

run_task(TaskConfig(
    model='your-model',
    datasets=['gsm8k'],
    analysis_report=AnalysisReportConfig(level='facts', language='en'),
))
```

Python `analysis_report=True` and CLI `--analysis-report` default to free `facts`, including future deliveries; analysis is otherwise off by default. Native evaluations launched from the Web enable facts automatically. `diagnose` and `full` require an explicitly configured analysis model, as described below.

Offline configuration accepts JSON or a JSON file:

```bash
evalscope analyze --run-dir outputs/20261116_120000 \
  --analysis-config '{"language":"en","weak_subset_min_samples":20,"trace":{"use_providers":false}}'
```

The Python entry points `evalscope.analysis.diagnose_run()` and `diagnose_benchmark()` use the same pipeline. The unified CLI entry point is `evalscope analyze`.

## Reading the results

Open the Web report's **Diagnostics** tab. Four cards show task failure rate, rule coverage, related components and analysis usage. The pattern table is followed by the subset heatmap and statistics. Hover for definitions and expand all statistics for distributions and details. Start with task failures and the known-outcome denominator. Excluded scores, unknown outcomes and execution errors without review records remain separate; missing data is neither zero score nor a pass.

- **Issue patterns** group observed types, related components and details. Shares use samples needing review, not the official failure denominator. Examples and the subset-by-type table open filtered predictions.
- **Facts** cover score distributions, weaker subsets, scoring statuses, judge coverage, recorded performance and token usage; empty generations, missing extracted answers, truncation, refusal text and output-length groups; single-letter choice distributions and confusion pairs; outcome stability of complete repeat groups; and Agent tool/trace findings.
- **Sample observations** appear in predictions, with filters for issue type, related component and finding category. Step badges require recorded tool-call IDs; unknown locations are not guessed.

Truncation uses prediction `stop_reason` first, then token-budget proximity only when the reason is missing and both token usage and the budget are available. Unavailable checks are listed as skipped. Continuous scores without an explicit threshold may have unknown outcomes. Weaker subsets require declared metric direction and bounded range.

Rates with events at or above 10% receive a warning by default (`warning_rate`). Weaker subsets must trail the overall result by at least 10 percentage points with at least 20 samples (`weak_subset_gap`, `weak_subset_min_samples`). Other facts are informational; P1 does not infer critical severity. Choice statistics require at least 90% single-uppercase-letter targets. Multiple-choice sets or full option-text answers are currently skipped; without an option mapping, letter membership is not guessed.

The taxonomy is extensible and cannot enumerate every failure cause. Failed samples without a matching rule remain visible. Correct answers may still involve tool errors or risk findings. Tool-derived arguments and state-changing calls alone do not establish policy violations. An empty extracted answer does not prove a correct original answer or a judge bug. A search with no results remains a tool observation; it does not establish invalid arguments. **An observed issue is not a proved failure cause.** Optional model diagnoses are proposals for review, as described below.

## Artifacts and cache

```text
<run>/analysis/<model>/<benchmark>/diagnosis.json
<run>/analysis/<model>/<benchmark>/samples.jsonl
<run>/reports/<model>/<benchmark>.json  # only analysis is updated
<run>/reports/report.html
```

Outputs include definitions, coverage, representative samples, rule findings and analysis usage (zero for facts). Exported evidence is size-limited and redacted; raw evaluation outputs are not copied into diagnostics.

Cache identity includes reviews, corresponding predictions, report metadata and consumed configuration, excluding the analysis writeback itself. Changed inputs or sample artifacts mark diagnostics stale; stale records cannot navigate into current predictions. Scores are unchanged, and an online analysis exception cannot fail a completed evaluation. Details and HTML retain legacy `Report.analysis` summaries when no diagnostic artifact exists.

## Model analysis, evidence and statistical scope

`analysis_report=True` still defaults to free `facts`. `diagnose` analyzes selected samples;
`full` also generates a referenced summary. Both require an explicit analysis model and remain experimental.
The evaluation or scoring model is never selected implicitly. Set `include_trace=False` to send only the task and final answer instead of intermediate messages and tool observations.

```python
from evalscope.api.analysis import AnalysisReportConfig

analysis = AnalysisReportConfig(
    level='full',
    model={
        'model_id': 'qwen3.8-max',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': 'env:DASHSCOPE_API_KEY',
    },
    max_diagnosed_samples=50,
    seed=42,
)
```

For offline analysis, save the same configuration as JSON:

```bash
evalscope analyze --run-dir outputs/<run> --analysis-config analysis.json
```

Evidence has two separate checks:

- **Locatability:** a source ID must exist in the redacted, truncated content actually sent to the model.
  A unique quotation can determine its range automatically. Explicit ranges use zero-based Unicode character
  offsets, with an exclusive end, and must match the quotation. Only whitespace is normalized. Missing,
  ambiguous, out-of-bounds or mismatched references are invalid. A proposed root step must have a valid
  reference on that exact supplied step.
- **Semantic support:** a locatable quotation does not establish that it supports a diagnosis. Independent
  annotation and review are still required. Responses with no usable evidence, failed requests, invalid
  formats or low confidence do not count as explained. Official scores and reference answers remain unchanged;
  suspected reference errors are only marked for review.

Denominators remain separate: official failure rates use known task outcomes; unavailable scores and safety
signals are recorded independently. Selection uses disjoint cohorts in this priority order: task failure,
unavailable scoring, safety signal, then process findings. Each cohort is stratified by subset, with at least
one sample per stratum. A budget too small to cover every stratum raises an error. High-confidence rule findings
do not establish causes and do not exclude official failures from selection.

Pattern counts describe current sample assignments, including unchanged rule assignments on unselected samples.
Population estimates are reported separately and weighted by each stratum's inclusion probability. The estimated
unresolved proportion among failed tasks uses only task-failure strata. Request failures and unusable diagnoses
remain unresolved. Sampling bounds combine stratum Wilson intervals with a Bonferroni correction, rather than
applying a pooled Wilson interval to quota samples. Census strata use exact proportions. These bounds describe
sampling uncertainty, not the probability that model explanations are correct.

Requests are cached by model, generation settings, protocol, prompt and exact supplied content. Interrupted runs
reuse completed requests. Invalid responses and transport failures are not cached as successful results.
`--force` issues new model requests. Reports distinguish current requests from cache reuse. Transport retries
belong to the existing model layer; the analyzer adds none. Request counts exclude provider-internal retries;
token usage comes from actual responses and missing usage is recorded separately. Check the provider's bill for
monetary costs.

Compare a sample file with independently prepared JSONL labels:

```bash
python -m evalscope.analysis.validation --samples <samples.jsonl> --labels <labels.jsonl> --output validation.json
```

Labels follow `DiagnosticLabel`; each file covers one run/model/benchmark. `rule_decisions` records explicit
positive and negative detector labels. An unannotated detector is not assumed negative. The tool reports type
agreement, attribution confusion, root-step agreement, citation locatability and reviewed evidence support
separately. Preserve annotator identity: agreement with model-assisted labels is not human-validated accuracy.
