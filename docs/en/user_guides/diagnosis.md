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

Python `analysis_report=True` and CLI `--analysis-report` default to free `facts`, including future deliveries; analysis is otherwise off by default. Native evaluations launched from the Web enable facts automatically. `diagnose` requires an explicitly configured analysis model, as described below.

Offline configuration accepts JSON or a JSON file:

```bash
evalscope analyze --run-dir outputs/20261116_120000 \
  --analysis-config '{"language":"en","weak_subset_min_samples":20,"trace":{"use_providers":false}}'
```

The Python entry points `evalscope.analysis.diagnose_run()` and `diagnose_benchmark()` use the same pipeline. The unified CLI entry point is `evalscope analyze`.

## Reading the results

Open **Diagnosis** in the Web report to generate or update basic analysis on saved results. To run model analysis for the current dataset, configure an OpenAI-compatible endpoint and model, choose the sample scope and limit, and start it explicitly. The form shows the maximum number of model requests and lets you omit intermediate steps and tool output. Web analysis supports up to 50 samples per request; use the CLI or Python for other configurations.

The Web reuses the saved basic-analysis settings, including score thresholds, for both levels. When the basic settings change, the report shows the updated facts until model analysis uses the same settings. For older artifacts whose settings cannot be recovered, run `evalscope analyze` once with the original configuration.

When scores are available without a pass threshold, **Review sample scores** shows the distribution and links to samples prioritized by the primary metric's direction. Diagnosis links preserve the issue, sample position and referenced step.

Diagnostics read saved evaluation records without requiring a benchmark adapter. A mean metric declared as accuracy or exact match can provide binary outcomes even if its score key uses a different name. Optional benchmark providers add benchmark-specific signals such as a safety verdict; they are not required for basic analysis.

Reports use three tabs: **Results → Diagnosis → Samples**. The shared dataset navigator shows each dataset’s metric, score and sample count. Switching tabs preserves the dataset; switching datasets clears old subset, diagnostic filters and sample selection. Narrow screens use a dataset dropdown. Token counts use compact K/M notation with exact values in tooltips. Latency/token details and task configuration are expanded by default and can be collapsed.

The **Diagnosis** tab starts with failed tasks and their known-outcome denominator. Open an issue to inspect its samples alongside the list. Unscored samples, unknown outcomes, and execution errors without records are shown separately. Analysis information shows the scope and, for model analysis, usage and citation checks.

- **Issue patterns** group visible observations among failed or unscored samples. A sample may appear in more than one group. Select an issue to inspect its matching samples and any model proposal.
- **Facts** cover score distributions, weaker subsets, scoring statuses, judge coverage, recorded performance and token usage; empty generations, missing extracted answers, truncation, refusal text and output-length groups; single-letter choice distributions and confusion pairs; outcome stability of complete repeat groups; and Agent tool/trace findings.
- **Sample observations** appear in Samples, with filters for issue type, related component and finding category. Issue entries open the matching sample collection. Use the sample header to open that exact record in the Samples tab. Sample links survive refresh and can be opened directly; a return action leads back to Diagnosis. Step badges require recorded tool-call IDs; unknown locations are not guessed.

The output-limit rate uses only predictions with a recorded ending reason. An answer can be complete even when the output limit is reached. Unavailable checks are listed as skipped. Continuous scores without an explicit threshold may have unknown outcomes. Weaker subsets require declared metric direction and bounded range.

Task outcomes use explicit benchmark results first, configured thresholds next, and then the main score’s recorded 0/1 result for binary accuracy or exact match. Continuous auxiliary scores do not invalidate a binary main score. Continuous scores without a pass threshold remain unknown and are excluded from the default failed-task scope. Set `trace={'outcome_thresholds': {'your_benchmark': {'quality': 0.8}}}` to declare a threshold, using the threshold metric’s declared direction (or an explicit `outcome_directions` override); alternatively use `scope='all'` or explicit `samples`. Nontext outputs are excluded from empty-text, answer-extraction and text-length statistics rather than mislabeled as empty answers.

Rates with events at or above 10% receive a warning by default (`warning_rate`). Weaker subsets must trail the overall result by at least 10 percentage points with at least 20 samples (`weak_subset_gap`, `weak_subset_min_samples`). Other facts are informational; P1 does not infer critical severity. Choice statistics require at least 90% single-uppercase-letter targets. Multiple-choice sets or full option-text answers are currently skipped; without an option mapping, letter membership is not guessed.

The taxonomy is extensible and cannot enumerate every failure cause. Failed samples without a matching rule remain visible. Correct answers may still involve tool errors or risk findings. Tool-derived arguments and state-changing calls alone do not establish policy violations. An empty extracted answer does not prove a correct original answer or a judge bug. A search with no results remains a tool observation; it does not establish invalid arguments. **An observed issue is not a proved failure cause.** Optional model diagnoses are proposals for review, as described below.

## Artifacts and cache

Basic and model analysis are stored separately. Updating basic analysis preserves current model proposals. If the source records change, regenerate basic analysis to see fresh observations and run model analysis again when needed.

```text
<run>/analysis/<model>/<benchmark>/diagnosis.json
<run>/analysis/<model>/<benchmark>/samples.jsonl
<run>/analysis/<model>/<benchmark>/diagnose/diagnosis.json
<run>/analysis/<model>/<benchmark>/diagnose/samples.jsonl
<run>/reports/<model>/<benchmark>.json  # only analysis is updated
<run>/reports/report.html
```

Outputs include definitions, coverage, representative samples, rule findings and analysis usage (zero for facts). Exported evidence is size-limited and redacted; raw evaluation outputs are not copied into diagnostics.

Cache identity includes reviews, corresponding predictions, report metadata and consumed configuration, excluding the analysis writeback itself. Changed inputs or sample artifacts mark diagnostics stale; stale records cannot navigate into current predictions. Scores are unchanged, and an online analysis exception cannot fail a completed evaluation. Diagnosis and HTML retain legacy `Report.analysis` summaries when no diagnostic artifact exists.

## Model analysis, evidence and statistical scope

`analysis_report=True` still defaults to free `facts`. `diagnose` includes basic analysis, model analysis of selected samples, citation checks, and a referenced summary with suggested checks. It requires an explicit analysis model and remains experimental.
The evaluation or scoring model is never selected implicitly. Set `include_trace=False` to send only the task and final answer instead of intermediate messages and tool observations.

```python
from evalscope.api.analysis import AnalysisReportConfig

analysis = AnalysisReportConfig(
    level='diagnose',
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
  annotation and review are still required. If any submitted citation fails, the sample proposal is
  withheld as a whole. Explicit scoring exclusion reasons and execution errors are shown directly
  from the record (with secrets redacted) in either mode. These samples do not use the model or take
  a sample budget slot. Default failed-task selection also skips samples whose only recorded issue is
  a generation budget stop and whose score explicitly records an empty extracted answer.
  Use an explicit sample selection or `scope='all'` to investigate those further.
  A model summary is requested only when at least one usable sample proposal can be included; rule facts are already in the report.
  Failed requests, invalid formats and low confidence also prevent proposals from being used. Official scores and reference answers remain unchanged;
  suspected reference errors are only marked for review.

Facts always cover all readable samples, independently of the model-analysis budget. Rule observations and model hypotheses remain separate; hypotheses never replace observed issue groups. Samples can have multiple observations, so group counts must not be added. Citation validation establishes locatability, not confirmed causes. No population cause proportions or unresolved-rate estimates are produced.

Model analysis defaults to failed tasks only. Set `scope` to `failed` (default), `unscored`, `risk` `process` (passing tasks with process issues), or `all` (every sample). Alternatively, `samples=[{'subset': 'main', 'index': 12}]` selects explicit identities and overrides scope. Missing requested samples fail before model requests.

`max_diagnosed_samples` is a cost-protection limit per benchmark: a positive integer caps the count; Python `None` / JSON `null` analyzes every eligible sample in the chosen scope. The default 50 is neither a quality guarantee nor a run-wide budget. Selection rotates across subsets and still runs when there are more subsets than slots. Reports show selected and eligible counts without extrapolating causes to unreviewed samples.

```bash
# Requires the explicit model configuration in analysis.json
evalscope analyze --run-dir outputs/<run> --analysis-config analysis.json --max-samples 100
evalscope analyze --run-dir outputs/<run> --analysis-config analysis.json --max-samples all
```

Inputs are retained completely when they fit `max_input_chars`. Otherwise the task, final answer and finding context take priority. Omitted context is recorded: absence in an excerpt cannot establish a missing operation or an unrecovered error.

With `include_trace=True`, saved, unredacted reasoning is supplied as a separate citable source linked to its original message/step. It remains model-generated evidence, not independent fact. Images, audio, video, internal data and protected reasoning are not sent to the diagnostic model; affected samples explicitly record incomplete context. Hidden reasoning is not recovered, and missing media cannot support inferred content.

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

In the Web dashboard, **Analysis information** reflects the mode: `facts` (basic analysis) shows the complete rule-statistics scope, no model usage, and generation time; `diagnose` (model analysis) adds the actual selection scope, selected samples, citation checks, model, new requests, cache reuse, and summary status. Summaries consider every valid hypothesis by primary type and count, then choose representative evidence across types within `max_input_chars`, independently of the display limit `max_examples`. Omitted explanations are recorded. Counts describe only model proposals in inspected samples, never population causes. Up to three observations and three suggested next checks appear separately; a statistic reference opens its supporting data. A locatable citation does not confirm a cause. There is no separate entry for process observations in passing samples.

Diagnosis lists show issue names and counts. “Other failed samples” means no known record anomaly matched; a model proposal may still exist. Embedded samples use top navigation and a step selector while sharing the complete QA/trace reader with Samples. Explanations cite specific subsets, samples and evidence-backed steps; opening Samples preserves the step. Statistical references expand their supporting data and denominator. Analysis information remains available. Older group citations are marked stale rather than guessed into sample references; rerun `evalscope analyze` to rebuild diagnostics. `facts` remains free of model calls.

Change older experimental configurations from `level='full'` to `level='diagnose'` and regenerate old diagnostic artifacts. Sample analyses and summaries are cached separately; a failed summary preserves completed sample analyses so a resumed run can reuse them.
