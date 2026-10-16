# Evaluate decision models with EvalScope: from quality to performance

An agent is about to stop. One small decision determines what happens next.

Is the task actually complete?

Similar decisions appear when it chooses a search tool, checks whether evidence is sufficient, or rates a result. A wrong choice, an unreliable probability, or a growing queue can affect every step that follows.

Jev brought attention to decision models. Once you have a model, practical questions remain: which judgments it handles well, whether its probabilities are useful, and how long decisions take under business traffic.

This guide follows the **EvalScope evaluation workflow**. ModelScope provides models and public datasets; EvalScope organizes requests, scoring, reports and performance tests. Connect an existing decision service or deploy open Clef-Flash, keeping the same System One evaluation configuration when switching services.

> Image placeholder 1 | ModelScope supplies models and datasets, an inference service returns decisions, and EvalScope handles quality evaluation, reports and performance tests. Show the role of each component.

## 1. Run three kinds of decisions

An agent needs current information. Choosing its next tool is **Choice**. Checking whether the available material is sufficient is **Noul**. Rating completion against an ordered rubric is **Score**.

Choice returns probabilities over candidates, Noul returns the probability that a condition holds, and Score returns probabilities over ordered levels. The expected Score can fall between two levels.

EvalScope preserves these outputs to evaluate both answers and probabilities. This guide evaluates individual text decisions. Tool execution and complete agent success require separate evaluation.

### Install EvalScope

Run these commands from an [EvalScope source checkout](https://github.com/modelscope/evalscope) containing the decision-evaluation features described here. In each new terminal, enter the same directory and activate the environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[perf,service]'
```

### Connect a decision service

EvalScope needs a model name, service address and API key. The connection example uses Bailian's hosted `decision-model-preview`; use an available model/address for your account and consult the [service reference](https://help.aliyun.com/zh/model-studio/decision-model-api) for region, workspace and access details. Replace these three values for an existing compatible service. To serve an open model first, follow section 3.

```bash
export DECISION_MODEL='decision-model-preview'
export DECISION_API_URL='https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
export DECISION_API_KEY='replace-with-your-service-key'
```

Save this as `evaluate.py` and run `python evaluate.py`.

```python
import os

from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model=os.environ['DECISION_MODEL'],
    eval_type='systemone_api',
    api_url=os.environ['DECISION_API_URL'],
    api_key=os.environ['DECISION_API_KEY'],
    datasets=['arc', 'boolq', 'stsb'],
    dataset_args={
        'arc': {'subset_list': ['ARC-Challenge']},
        'boolq': {'extra_params': {'decision_kind': 'noul'}},
    },
    generation_config={'timeout': 300, 'retries': 1},
    eval_batch_size=1,
    limit=5,
))
```

One run covers all three kinds: ARC-Challenge for Choice, BoolQ for Noul, and STS-B for similarity ratings on a 0–5 scale. Datasets load from ModelScope by default. Gold answers are used for grading and are never sent to the service.

Keep `limit=5` for the first run, then remove it for a complete evaluation. The limit applies per subset, so a benchmark with multiple subsets may send more than five requests.

`retries=1` means one total attempt per sample. Do not add temperature, max_tokens or streaming controls to the decision configuration.

## 2. Read the report before comparing models

After evaluation, view the results in the EvalScope Web UI. For a source install, install Node.js and build the frontend once. From the repository root, run:

```bash
make web-build
evalscope service --host 127.0.0.1 --port 9000 --outputs ./outputs
```

Open `http://127.0.0.1:9000`, choose Evaluations and open the model report. Details shows metrics and lets you select the decision group, confidence source and accepted coverage to inspect risk. Predictions supports dataset switching, sample filtering and per-sample probability distributions, with original and scoring probability views.

For each sample, the probability panel shows the prediction, reference and option probabilities. Evaluation Result shows the score. Raw scoring data stays collapsed until you need it for troubleshooting.

The output directory also keeps `predictions/`, `reviews/` and the HTML summary for later inspection.

Check successful requests, valid review counts and coverage before reading scores. A high score on a partially completed test does not represent the whole benchmark.

> Image placeholder 2 | A real report showing the primary metric, valid review count and sample detail entry. Add the screenshot after running the evaluation.

Begin with three groups of metrics.

**Were the answers correct?** Accuracy measures correct Choice/Noul answers. Fixed-class tasks can also use macro-F1 to give every class equal weight.

**Were the probabilities useful?** NLL and Brier penalize confident mistakes; lower is better. ECE groups predictions by probability and checks whether, for example, predictions near 80% are correct about 80% of the time. It describes a collection of predictions, not a guarantee for one answer. Read it alongside accuracy.

**How far were ratings from the reference?** Score MAE measures expected-rating error. On STS-B's 0–5 scale, MAE 0.6 means an average error of about 0.6 points. Continuous gold is preserved, so this adapter does not round it for classification accuracy.

If your application accepts confident answers and sends other cases for review, inspect AURC and risk–coverage data. They show the trade-off between retained coverage and error risk as the acceptance threshold changes. Option probabilities and an additional provider `confidence` can have different meanings; the report distinguishes them.

> Image placeholder 3 | An illustration of similar accuracy with different probability quality, alongside a risk–coverage illustration. Label both as illustrative data.

Use these auxiliary metrics when needed.

| Metric | What it shows | Better direction |
|---|---|---|
| macro-F1 | Fixed-class performance with equal class weighting | Higher |
| Precision / Recall / F1 | Reliability of positive Noul predictions, positive coverage and their balance | Higher |
| AUROC / AUPRC | Separation of positive and negative Noul samples | Higher; compare datasets cautiously |
| Spearman / Pearson | Agreement in rating order and score variation | Closer to 1 |
| nRPS | Probability concentrated near the correct discrete Score level | Lower |

Small samples, a single gold class or constant ratings can leave metrics undefined. Inspect the reason instead of interpreting an absent value as zero. Avoid directly averaging binary/multiclass Brier or errors on different Score scales.

## 3. Download an open model from ModelScope and serve it

For self-hosting, use [Cloudflare/clef-flash](https://modelscope.cn/models/Cloudflare/clef-flash) on ModelScope. Set `VLLM_USE_MODELSCOPE=true` so vLLM downloads and loads it automatically, then connect EvalScope to evaluate it.

Clef-Flash is an approximately 9B-parameter model based on Qwen3.5-9B with a joint decision head. This guide uses text inputs.

The commands target Linux/NVIDIA GPUs with sufficient memory. The [latest vLLM development documentation](https://docs.vllm.ai/en/latest/models/structured_decisions_models/) lists native Clef-Flash System One support. Install a development build containing that implementation; loading the backbone alone does not establish decision-head support.

```bash
pip install -U modelscope
pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
VLLM_USE_MODELSCOPE=true vllm serve Cloudflare/clef-flash \
  --served-model-name clef-flash \
  --max-model-len 16384 --port 8000
```

Keep that terminal running. In another activated terminal, change the connection settings.

```bash
export DECISION_MODEL='clef-flash'
export DECISION_API_URL='http://127.0.0.1:8000/v1'
export DECISION_API_KEY='EMPTY'
```

Check one decision before starting evaluation.

```bash
curl "$DECISION_API_URL/systemone" \
  -H 'Content-Type: application/json' \
  -d '{"model":"clef-flash","state":"The sky is blue.","questions":{"answer":{"type":"choice","instructions":"What color is the sky?","criteria":{"blue":"Blue","red":"Red"}}}}'
```

The response should contain `answers.answer.choice` and `probabilities` keyed by option ID. Values depend on the actual response.

Run `python evaluate.py` again. The model name, address and key changed; the same `systemone_api` evaluation configuration still applies. `EMPTY` denotes that this local example has no configured authentication.

**Deployment verification scope:** these commands follow the current vLLM development documentation and have not been validated by this project on NVIDIA GPUs. Earlier real Clef-Flash quality runs used the official PyTorch implementation. Those are separate kinds of evidence. Record model revision, vLLM version and serving arguments for comparisons, and check whether long inputs are truncated.

## 4. Choose evaluation data from ModelScope

Start with the three introductory datasets. EvalScope maps benchmark names to ModelScope datasets, builds task-specific requests, reads gold answers and computes metrics. Change the dataset configuration to extend your evaluation.

| Benchmark | What it measures |
|---|---|
| `arc` | Scientific knowledge and reasoning; this guide uses ARC-Challenge |
| `boolq` | Yes/no judgments based on a passage; this guide uses Noul |
| `stsb` | Sentence-pair semantic similarity, rated from 0 to 5 |
| `jevbench` | Choice, Noul and Score decisions across 231 public tasks |
| `open_decisions` | Three decision types across 2,352 tasks, including Intelligence, Calibration and Capability |
| `banking77` | Banking intent recognition across 77 classes |
| `reward_bench` | Choosing the better of two answers |
| `general_decision` | Choices, boolean judgments and ratings from your own application |

[JevBench](https://modelscope.cn/datasets/evalscope/jevbench) and [OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions) are available on ModelScope. See the detailed usage documentation linked at the end for sources, scoring definitions and service requirements.

Replace the dataset settings in your working `evaluate.py` to try the comprehensive decision sets.

```python
datasets=['jevbench', 'open_decisions'],
dataset_args={},
```

Keep the limit for the smoke run, then remove it. Public decision scores do not represent complete agent success. RewardBench's sample-weighted score is also different from its official category-weighted leaderboard result.

## 5. Bring the decisions from your application

Return to the agent at the start. Turn its actual routing choices, completion checks and rating rubrics into evaluation records.

Create `decision-data/default_test.jsonl` with one question per line. These three records are illustrative inputs.

```json
{"id":"route","state":"The user needs today's weather. No online result has been obtained.","question":{"type":"choice","instructions":"Which capability should run next?","criteria":{"search":"Look up external information","calculate":"Perform a numerical calculation","answer":"Answer from available information"}},"expected":"search"}
{"id":"ready","state":"The user needs today's weather. No online result has been obtained.","question":{"type":"noul","instructions":"Has today's weather information already been obtained?"},"expected":"no"}
{"id":"complete","state":"The task requires price, stock availability and a purchase link. The answer gives only the price.","question":{"type":"score","instructions":"Rate completion by the number of requirements covered.","criteria":["None covered","One covered","Two covered","All three covered"]},"expected":"1"}
```

`state` supplies the material; `question` defines the task; `expected` is the reference used only for grading. Choice uses an option ID, Noul uses `yes` / `no`, and Score uses a zero-based level number. Define real business criteria clearly before obtaining human or reliable rule-based references.

Replace the dataset settings in the same script. Local records do not need to be uploaded.

```python
datasets=['general_decision'],
dataset_args={'general_decision': {'local_path': './decision-data'}},
```

These three records check formatting and execution. Use independent, representative samples for conclusions about quality, including ambiguous and insufficient-evidence cases.

For all supported fields, multiple subsets and continuous reference scores, see [Custom decision datasets](../advanced_guides/custom_dataset/decision.md).

## 6. Check what happens when requests arrive together

Quality evaluation measures decisions; performance testing measures the wait for a decision and the number completed each second.

Stop quality-evaluation traffic on the same service. Fix the model configuration and inputs. These commands use the active terminal's `DECISION_*` settings for a compatible hosted or local inference service. Respect hosted quotas and begin with a small run.

Export ARC-Challenge Choice, BoolQ Noul and STS-B Score inputs. Each file contains 20 distinct requests by default; performance tests cycle through them.

```bash
python examples/decision/prepare_perf_requests.py \
  --api systemone --model "$DECISION_MODEL" \
  --choice-benchmark arc --output ./perf-input/systemone
evalscope perf --api systemone --model "$DECISION_MODEL" \
  --url "$DECISION_API_URL" --api-key "$DECISION_API_KEY" \
  --dataset line_by_line --dataset-path ./perf-input/systemone/choice.jsonl \
  --parallel 1 --number 20 --warmup-num 2 --no-stream \
  --read-timeout 300 --outputs-dir ./perf-results/smoke
```

Once that works, test each kind across concurrency levels.

```bash
for kind in choice noul score; do
  evalscope perf --api systemone --model "$DECISION_MODEL" \
    --url "$DECISION_API_URL" --api-key "$DECISION_API_KEY" \
    --dataset line_by_line --dataset-path "./perf-input/systemone/$kind.jsonl" \
    --parallel 1 4 8 16 --number 1000 1000 1000 1000 \
    --warmup-num 20 --no-stream --read-timeout 300 \
    --outputs-dir "./perf-results/$kind"
done
```

Each kind uses concurrency 1/4/8/16, with 20 warmups and 1,000 measured requests per setting. Export again after switching models so requests do not retain the previous model name.

In the Web UI Performance page, select the `perf-results` directory and open a run to switch charts, inspect concurrency results and browse request records. The output directory also keeps `perf_report.html`. Check success rate and failures, then p50/p90/p95/p99 latency and successful requests/s. A p95 of 0.3 seconds means about 95% of successful requests finish within that time. One request is one question here, so successful requests/s equals successful decisions/s.

If concurrency raises latency without improving throughput, requests are probably queueing. Network, request length, caching and measurement location also affect comparisons.

> Image placeholder 4 | Successful throughput and p95 latency versus concurrency for one model, with input lengths and cache/serving settings. Add measured results later.

An invalid decision fails even with HTTP 200. Request latency includes HTTP network, queueing and service execution, rather than model compute alone. These non-streaming decisions do not use TTFT/TPOT/ITL. See the [performance quick start](../user_guides/stress_test/quick_start.md) for additional options.

## 7. Customize with ms-swift when needed

[ms-swift](https://swift.readthedocs.io/en/latest/BestPractices/Decision.html) provides decision training for JEV-27B-VL, Cloudflare/clef, OmniJev and generic CausalLMs. Trained models can run through Transformers; a compatible decision-service adapter can then connect them to EvalScope for re-evaluation.

`Cloudflare/clef` and `clef-flash` are different models. Exporting a trained ms-swift checkpoint to vLLM System One requires separate adaptation and validation; this guide does not claim that complete training-to-serving path has been verified. Keep training, calibration and evaluation data separate.

## Troubleshooting

| Symptom | Check first |
|---|---|
| `401` / `403` or unavailable model | API key, region, workspace and model access |
| `404` / `501` | `/v1/systemone` path and a vLLM build with the model's decision strategy |
| Too many Choice options | Actual service limit; BANKING77 requires all 77 options |
| Unsupported Score levels | Preserve the original rubric and use a compatible service |
| Scores with incomplete review counts | Failed/invalid responses, coverage and saved predictions/reviews |
| Want a small first run | Keep `limit=5`; use 20 requests for perf |

## Other interfaces

[TypeSafe Jev](https://api.typesafe.ai/redoc) also uses `systemone_api`. Set the address to `https://api.typesafe.ai/v1`, choose a model returned by your account's `/v1/models` and provide its API key. The EvalScope benchmark, scoring and performance-testing workflow stays the same; check each model's capabilities and limits.

AutoTrust JEV's `/v1/decide` uses `autotrust_api`; see `examples/decision/serve_autotrust.py`. Its fixed-six-level Score cannot cover complete JevBench/OpenDecisions. Ordinary chat models use `openai_api` and can run ARC, BANKING77, default-Choice BoolQ and RewardBench for answer correctness. Self-reported confidence in chat does not substitute for option probabilities.

## Continue with EvalScope

For more options and full instructions, see:

- [EvalScope documentation](https://evalscope.readthedocs.io/en/latest/)
- [Decision API configuration](../user_guides/systemone.md)
- [Supported benchmarks](../get_started/supported_dataset/llm.md)
- [Web service and result browsing](../user_guides/service.md)
- [Performance quick start](../user_guides/stress_test/quick_start.md)
