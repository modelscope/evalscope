# Decision Models

Use `general_decision` to evaluate choices, yes/no judgments and ratings from your own application. Records stay in local JSONL files; no dataset upload is needed.

This format requires `systemone_api` or `autotrust_api`. For deployment and connection settings, see the [decision evaluation best practice](../../best_practice/decision_evaluation.md). For ordinary chat-model questions, use the [LLM dataset formats](llm.md).

## Prepare a dataset

Create this directory where you will run evaluation:

```text
decision-data/
└── default_test.jsonl
```

Save these three records in `decision-data/default_test.jsonl`, one complete JSON object per line, using UTF-8. They cover Choice, Noul and a four-level Score:

```json
{"id":"route","state":"The user asks for today's weather. No online search result is available yet.","question":{"type":"choice","instructions":"Which capability should be used next?","criteria":{"search":"Look up external information","calculate":"Perform a numerical calculation","answer":"Answer using existing information"}},"expected":"search"}
{"id":"ready","state":"The user asks for today's weather. No online search result is available yet.","question":{"type":"noul","instructions":"Has today's weather information already been obtained?"},"expected":"no"}
{"id":"complete","state":"The task asks for price, stock and a purchase link. The answer contains only the price.","question":{"type":"score","instructions":"Rate completion by the number of requested items covered.","criteria":["No items covered","One item covered","Two items covered","All three items covered"]},"expected":"1"}
```

These are format examples. Use independent, representative business cases with reliable reference answers for model comparisons.

### Required fields

| Field | Type | Meaning |
|---|---|---|
| `state` | JSON value, usually a string or object | Material available at the decision point |
| `question` | Object | One question per record |
| `question.type` | String | `choice`, `noul` or `score` |
| `question.instructions` | Nonempty string | The question and judgment criteria |
| `question.criteria` | Object or array | Required for Choice/Score; optional for Noul, as below |
| `expected` | Reference answer | Used only for scoring; meanings below |

`state` and `question` are sent to the service. Keep reference answers, correct-answer explanations and teacher predictions out of them.

### Question types

| Type | `criteria` | `expected` | Grading |
|---|---|---|---|
| Choice | Object mapping option IDs to descriptions, in the desired order | An option ID, e.g. `"search"`, not its description | Compare the highest-probability option with the reference; ties are resolved deterministically by option ID |
| Noul | Usually omitted; optionally an object describing `true` and/or `false` | `"yes"` / `"no"`; booleans and `"true"` / `"false"` are also accepted | Compare `P(yes)` with the classification threshold, 0.5 by default |
| Score | Level descriptions ordered from lowest to highest | A zero-based level, e.g. `"1"` or integer `1` | Argmax gives level accuracy; the distribution gives the probability-weighted score and rating errors |

The four Score descriptions above correspond to `0`, `1`, `2`, `3`. `expected="1"` means “One item covered,” not a one-based index. For normalized RPS, a distant wrong level costs more than a nearby one.

EvalScope accepts 2–255 Choice options and 2–10 Score levels; the service may have narrower limits. AutoTrust's native `/decide` requires **exactly six Score levels**. Do not pad a four-level rubric to change its meaning.

### Optional fields

| Field | Default | Meaning |
|---|---|---|
| `id` | Absent | Source identifier retained in metadata; use unique strings for traceability |
| `continuous` | `false` | Score only: a continuous reference within `0` to `number of levels - 1`; see below |
| `fixed_classes` | `false` | Enable macro-F1 for a fixed taxonomy; option IDs must keep the same meaning across records |
| `family` | Empty | For discrete references, add `family_macro`: average accuracy per family, then average families equally |
| `group`, `tier`, `language` | Absent / empty | Provenance retained in metadata; these do not automatically create subsets or metrics |

## Run evaluation

With a local System One service running, save this as `evaluate_custom.py` and run `python evaluate_custom.py`. For an existing service, set `DECISION_MODEL`, `DECISION_API_URL` and `DECISION_API_KEY`; the dataset stays the same.

```python
import os

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model=os.getenv('DECISION_MODEL', 'clef-flash'),
    eval_type='systemone_api',
    api_url=os.getenv('DECISION_API_URL', 'http://127.0.0.1:8000/v1'),
    api_key=os.getenv('DECISION_API_KEY', 'EMPTY'),
    datasets=['general_decision'],
    dataset_args={
        'general_decision': {
            'local_path': './decision-data',
        },
    },
    generation_config={'timeout': 300, 'retries': 1},
    eval_batch_size=1,
))
```

`local_path` points to the directory. The default subset is `default`, and the evaluation split is `test`, hence `default_test.jsonl`. To use `routing_test.jsonl` and `quality_test.jsonl`, set `subset_list=['routing', 'quality']` in the benchmark's dataset arguments. Each file may contain all three types.

No chat generation parameters or LLM judge are needed: references are graded directly against the returned decisions.

## Read the metrics

The default primary metric is **accuracy** across discrete-reference records. Other metrics are separated by type and, for Score, by level count. Do not average incompatible error scales.

| Type | Metrics | Interpretation |
|---|---|---|
| Choice | Accuracy; NLL, multiclass Brier, ECE; macro-F1 for fixed classes | Accuracy/F1: higher is better. Probability errors: lower is better |
| Noul | Accuracy, Precision, Recall, F1, AUROC, AUPRC; NLL, binary Brier, ECE | Classification/ranking: higher is better. Probability errors: lower is better |
| Discrete Score | Accuracy, MAE, normalized RPS; NLL, multiclass Brier, ECE; Spearman, Pearson | MAE uses level units. Errors: lower is better. Correlations closer to 1 are better |
| All types | AURC and saved risk–coverage points | Lower AURC is better; check the confidence source and risk definition |

NLL penalizes low probability on the reference; its numerical floor is saved as `nll_epsilon`. ECE uses 10 equal-width bins: `P(yes)` for Noul, top-label probability for Choice/discrete Score. Binary and multiclass Brier have different definitions and should not be averaged together.

Choice/Noul risk is the error rate among accepted predictions; Score risk is MAE divided by the score range. Top option probability and provider `confidence` are separate confidence sources. The Web UI coverage control selects saved points without splitting tied-confidence groups.

The three introductory records are too few for some metrics: AUROC may be unavailable, and Score correlations cannot be calculated. The report keeps the reason; an unavailable value is not zero. Check execution coverage and valid counts before comparing models.

View results through the [EvalScope Web UI](../../best_practice/decision_evaluation.md#2-read-the-report-before-comparing-models), or inspect `predictions/`, `reviews/` and `reports/`. The probability panel switches between original and scoring distributions; reviews record any normalization of rounded probabilities.

## Continuous reference scores

For a real-valued reference, set `continuous=true`. Save this record separately as `decision-data/continuous_test.jsonl`:

```json
{"id":"similarity","state":{"sentence1":"A girl is styling her hair.","sentence2":"A girl is brushing her hair."},"question":{"type":"score","instructions":"Rate the semantic similarity of the two sentences.","criteria":["Unrelated","Some shared meaning","Mostly similar","Equivalent"]},"expected":1.5,"continuous":true}
```

To evaluate only this four-level continuous subset, replace `dataset_args` in the script with:

```python
dataset_args={
    'general_decision': {
        'local_path': './decision-data',
        'subset_list': ['continuous'],
        'metric_list': ['mae'],
        'primary_metric': 'mae',
    },
},
```

MAE is now the primary metric. Continuous references are not rounded to calculate Accuracy, NLL, Brier, ECE or RPS. MAE, correlations and risk–coverage remain available when their sample requirements are met. Convert business ratings on another scale to the declared zero-based scale first.

For more options, see [Decision API configuration](../../user_guides/systemone.md) and [evaluation parameters](../../get_started/parameters.md).
