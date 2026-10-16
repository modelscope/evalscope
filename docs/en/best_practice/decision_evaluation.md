# Start with a wrong answer: evaluate decision models with EvalScope

In a real BoolQ trial run, a decision model assigned **86% probability to Yes**.

The reference answer was No.

The selected answer tells us this prediction was wrong. Its probability tells us something else: it was a confident mistake. Both matter when an application uses probabilities to decide which action to take next.

Think of the small judgments inside an agent. Should it search or answer now? Has it obtained the information it needs? How many requirements has it completed? Each judgment may return just an option, a yes/no result or a rating, yet affect the actions that follow.

Decision models such as Jev and Clef-Flash return probabilities alongside these judgments. Evaluation therefore needs to check answer correctness, probability quality and, as traffic grows, response time.

This guide follows those three questions with EvalScope: run an evaluation, inspect a wrong answer, then test concurrent requests. ModelScope supplies public datasets and open models; the EvalScope Web UI lets you explore the results.

If you already have a decision API, start below. For an open model on your own machine, begin with [section 5](#5-serve-clef-flash-from-modelscope-with-vllm). The main workflow uses System One throughout, so changing services only requires updating connection settings.

> Image placeholder 1 | ModelScope supplies models and datasets, an inference service returns decisions, and EvalScope handles quality evaluation, reports and performance tests. Show the role of each component.

## 1. Run three kinds of decisions

Consider an assistant that needs current information. The three question kinds fit into one workflow.

| Judgment in the application | Kind | Model output |
|---|---|---|
| Search, calculate or answer next | Choice | Probabilities over the options |
| Has the required information been obtained | Noul | Probability that the condition holds |
| How many requirements have been completed | Score | Probabilities over ordered levels and their expected score |

The expected Score can be fractional, falling between two levels when probability is distributed across them.

We will use three public datasets to exercise all three paths. This evaluates individual text judgments. Tool execution and complete agent success require separate evaluation.

### Install EvalScope

Run these commands from an [EvalScope source checkout](https://github.com/modelscope/evalscope) containing the decision-evaluation features described here. In each new terminal, enter the same directory and activate the environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[perf,service]'
```

### Connect a decision service

Prepare a model name, service address and API key. This example uses DashScope's hosted `decision-model-preview`; consult the [service reference](https://help.aliyun.com/zh/model-studio/decision-model-api) for region, workspace and access details. Replace these three values for another System One service.

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

EvalScope loads the datasets from ModelScope, sends the questions and grades the responses. Reference answers are never sent to the model.

`datasets` selects the tasks; `eval_type='systemone_api'` selects the interface. ARC-Challenge covers Choice, BoolQ explicitly switches to Noul, and STS-B uses Score on a 0–5 scale. Keep these tasks and scoring settings when comparing models.

Keep `limit=5` to check that all three paths return valid results, then remove it for a complete evaluation. The limit applies per subset, so multiple subsets may send more than five requests. `retries=1` means one total attempt per sample. No temperature, max_tokens or streaming controls are needed.

## 2. Read the report before comparing models

After evaluation, view the results in the EvalScope Web UI. For a source install, install Node.js and build the frontend once. From the repository root, run:

```bash
make web-build
evalscope service --host 127.0.0.1 --port 9000 --outputs ./outputs
```

Open `http://127.0.0.1:9000`, choose Evaluations and open the model report. Check successful requests, valid review counts and coverage before reading scores.

A real trial with the configuration above produced these results for `decision-model-preview`. Each dataset has only five samples: these illustrate how to read the report, not overall model quality. A new run may produce different values.

| Dataset | Valid reviews | Primary metric | Interpretation |
|---|---|---|---|
| ARC-Challenge | 5 | Accuracy 80% | Four of five Choice answers correct |
| BoolQ in Noul mode | 5 | Accuracy 80% | Four of five yes/no answers correct |
| STS-B | 5 | MAE 0.564 | Average error of 0.564 points on a 0–5 scale |

In Predictions, switch datasets and filter for wrong answers to inspect your own results. In this BoolQ trial, the sample shows Yes at 86%, No at 14%, and No as the reference. The distribution makes this mistake easier to understand than an aggregate score alone.

Evaluation Result shows the sample score. Expand Raw scoring data or switch between original and scoring probabilities when investigating a response. The output directory also retains `predictions/`, `reviews/` and the HTML summary.

> Image placeholder 2 | The actual trial report and BoolQ error, highlighting valid review counts, Yes at 86% and the No reference. Use the corresponding saved run when adding images together.

That wrong answer gives the metrics a concrete purpose.

**Accuracy measures answer correctness.** Assigning Yes 51% or 86% both produces a wrong answer in this example. Accuracy cannot distinguish the two probabilities.

**NLL, Brier and ECE help assess probabilities.** For this same No reference, increasing the Yes probability from 51% to 86% worsens NLL and Brier. Both penalize confident errors; lower is better. ECE checks agreement between probabilities and observed frequencies. For Choice, selected answers assigned 80% probability should be correct about 80% of the time. For Noul, conditions assigned an 80% Yes probability should hold about 80% of the time. One wrong prediction does not establish miscalibration; five samples are too few to assess it reliably.

**MAE measures rating error.** STS-B references can be fractional. EvalScope compares the model's expected score with those references, preserving that detail. The table's 0.564 means an average error of 0.564 points; lower is better.

An application might accept some decisions automatically and send others for review. In Details, the risk–coverage interaction lets you select the question group and probability source, then change accepted coverage to inspect retained risk. AURC summarizes the curve; lower is generally better. Score risk uses rating error normalized by the scale range.

Option probabilities and a provider's separate `confidence` may have different meanings, so the report distinguishes them. Validate actual business thresholds on independent, representative data rather than this small trial.

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

## 3. Choose evaluation data from ModelScope

Once all three request kinds work, choose the tasks that matter to your application. Scientific reasoning, intent classification and broader decision making call for different datasets.

These benchmarks are already integrated. Use the name in the left column and EvalScope loads the data, builds questions and computes metrics.

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

## 4. Bring the decisions from your application

Public benchmarks provide a shared reference. To assess your own application, add the judgments it encounters in practice.

Use the assistant workflow from section 1: choosing a capability, checking whether information has been obtained, and counting completed requirements. Each can become a question with a clear reference answer.

Create `decision-data/default_test.jsonl` with one question per line. These three records are illustrative inputs.

```json
{"id":"route","state":"The user needs today's weather. No online result has been obtained.","question":{"type":"choice","instructions":"Which capability should run next?","criteria":{"search":"Look up external information","calculate":"Perform a numerical calculation","answer":"Answer from available information"}},"expected":"search"}
{"id":"ready","state":"The user needs today's weather. No online result has been obtained.","question":{"type":"noul","instructions":"Has today's weather information already been obtained?"},"expected":"no"}
{"id":"complete","state":"The task requires price, stock availability and a purchase link. The answer gives only the price.","question":{"type":"score","instructions":"Rate completion by the number of requirements covered.","criteria":["None covered","One covered","Two covered","All three covered"]},"expected":"1"}
```

Each record has three core parts. `state` supplies the material, `question` defines the judgment, and `expected` is the reference retained by the evaluator rather than sent to the model.

Choice uses an option ID, Noul uses `yes` / `no`, and Score uses a zero-based level number. The last example covers one requirement, so its reference is `1`. Clear criteria give you the same test when changing models, prompts or services.

Replace the dataset settings in the same script. Local records do not need to be uploaded.

```python
datasets=['general_decision'],
dataset_args={'general_decision': {'local_path': './decision-data'}},
```

These three records check formatting and execution. Use independent, representative samples for conclusions about quality, including ambiguous and insufficient-evidence cases.

For all supported fields, multiple subsets and continuous reference scores, see [Custom decision datasets](../advanced_guides/custom_dataset/decision.md).

## 5. Serve Clef-Flash from ModelScope with vLLM

Self-hosting is optional. To run on your own machine, use [Cloudflare/clef-flash](https://modelscope.cn/models/Cloudflare/clef-flash) on ModelScope. Based on Qwen3.5-9B, this approximately 9B-parameter model supports all three question kinds. This guide uses text inputs.

Once serving is ready, run the same `evaluate.py`. Keeping the datasets and scoring settings fixed makes hosted and self-hosted results easier to compare.

The commands target Linux/NVIDIA GPUs with sufficient memory. The [latest vLLM development documentation](https://docs.vllm.ai/en/latest/models/structured_decisions_models/) lists native Clef-Flash System One support. Install a development build containing that implementation; loading the backbone alone does not establish decision-head support.

```bash
pip install -U modelscope
pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
VLLM_USE_MODELSCOPE=true vllm serve Cloudflare/clef-flash \
  --served-model-name clef-flash \
  --max-model-len 16384 --port 8000
```

`VLLM_USE_MODELSCOPE=true` makes vLLM download the model from ModelScope automatically. No separate download script is needed. Keep the service terminal running and change the connection settings in another activated terminal.

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

## 6. Check what happens when requests arrive together

A quick response to one request does not tell you what happens when many arrive together. If a judgment sits on an agent's critical path, queueing adds directly to the user's wait.

Use EvalScope to measure how long a complete decision request takes and how many the service completes each second.

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

Each kind uses concurrency 1/4/8/16, with 20 warmups and 1,000 measured requests per setting. Export again after switching models so requests do not retain the previous model name. This is a reproducible test configuration; measure actual performance on your own service.

In the Web UI Performance page, select `perf-results` and open a run to compare concurrency charts and browse request records. The output directory also keeps `perf_report.html`.

Start with success rate and failures: fast responses with many failures do not establish a usable service. Then inspect p50/p90/p95/p99 latency and successful requests/s. For example, a p95 of 0.3 seconds means about 95% of successful requests finish within that time. One request is one question here, so successful requests/s equals successful decisions/s.

If concurrency raises latency without improving throughput, requests are probably queueing. Network, request length, caching and measurement location also affect comparisons.

> Image placeholder 4 | Successful throughput and p95 latency versus concurrency for one model, with input lengths and cache/serving settings. Add measured results later.

An invalid decision fails even with HTTP 200. Request latency includes HTTP network, queueing and service execution, rather than model compute alone. These non-streaming decisions do not use TTFT/TPOT/ITL. See the [performance quick start](../user_guides/stress_test/quick_start.md) for additional options.

## 7. Customize with ms-swift when needed

If errors cluster around a particular business judgment, the evaluation can help identify a training target. Keep the same independent test set after training to check whether those errors improved.

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

You can now inspect an answer and its probabilities, add business questions and compare response times under load. The opening's wrong answer with 86% probability is something you can locate, examine and test again after a model update.

For more options and full instructions, see:

- [EvalScope documentation](https://evalscope.readthedocs.io/en/latest/)
- [Decision API configuration](../user_guides/systemone.md)
- [Supported benchmarks](../get_started/supported_dataset/llm.md)
- [Web service and result browsing](../user_guides/service.md)
- [Performance quick start](../user_guides/stress_test/quick_start.md)
