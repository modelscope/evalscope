# From Jev to Clef-Flash: measure decision quality and performance with EvalScope

An agent is about to stop. One small decision determines what happens next.

Is the task actually complete?

Similar decisions appear when it chooses a search tool, checks whether evidence is sufficient, or rates a result. A wrong choice, an unreliable probability, or a growing queue can affect every step that follows.

Jev brought attention to these frequent judgments. TypeSafe offers structured decisions and probabilities that software can consume directly; open models such as Clef-Flash provide related capabilities. Whether they save time and suit your application is something you can measure.

Start with **Bailian's hosted API, then optionally deploy Clef-Flash using the same System One format**. After the public benchmarks work, bring your own decision data.

> Image placeholder 1 | An agent choosing tools and checking completion, with Choice, Noul and Score at the decision points. Mark this as an illustration; do not add unmeasured speed claims.

## 1. Run three kinds of decisions

An agent needs current information. Choosing its next tool is **Choice**. Checking whether the available material is sufficient is **Noul**. Rating completion against an ordered rubric is **Score**.

Choice returns probabilities over candidates, Noul returns the probability that a condition holds, and Score returns probabilities over ordered levels. The expected Score can fall between two levels.

EvalScope preserves these outputs to evaluate both answers and probabilities. This guide evaluates individual text decisions. Tool execution and complete agent success require separate evaluation.

### Install EvalScope

Run these commands from an [EvalScope source checkout](https://github.com/modelscope/evalscope) containing the decision-evaluation features described here. In each new terminal, enter the same directory and activate the environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[perf]'
```

### Connect to Bailian

Prepare a Bailian API key with access to `decision-model-preview`. The example uses the Beijing trial endpoint. Use the address for your own region/workspace when required by the [Bailian API reference](https://help.aliyun.com/zh/model-studio/decision-model-api).

```bash
export DASHSCOPE_API_KEY='replace-with-your-Bailian-key'
export DECISION_API_KEY="$DASHSCOPE_API_KEY"
export DECISION_API_URL='https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
export DECISION_MODEL='decision-model-preview'
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

The terminal prints the output directory. Open `outputs/<timestamp>/reports/report.html` for the summary. Inspect individual predictions and reviews in the same run's `predictions/` and `reviews/` directories.

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

## 3. Keep the interface and serve an open model

After the hosted run works, you can use a GPU to serve [Cloudflare/clef-flash](https://modelscope.cn/models/Cloudflare/clef-flash) yourself.

Clef-Flash is an approximately 9B-parameter model based on Qwen3.5-9B with a joint decision head. This guide uses text inputs.

The commands target Linux/NVIDIA GPUs with sufficient memory. The [latest vLLM development documentation](https://docs.vllm.ai/en/latest/models/structured_decisions_models/) lists native Clef-Flash System One support. Install a development build containing that implementation; loading the backbone alone does not establish decision-head support.

```bash
pip install -U uv modelscope
uv pip install -U vllm --torch-backend auto \
  --extra-index-url https://wheels.vllm.ai/nightly
modelscope download --model Cloudflare/clef-flash \
  --local_dir ./models/clef-flash
vllm serve ./models/clef-flash \
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

## 4. Choose benchmarks that match your goal

Start with the three introductory datasets, then extend the evaluation.

| Benchmark | Task | What to check |
|---|---|---|
| `arc` | Science multiple-choice questions | Select `ARC-Challenge` here |
| `boolq` | Yes/no reading comprehension | Choice by default; this guide enables Noul with threshold 0.5 |
| `stsb` | Continuous sentence-similarity ratings | Six output levels, continuous gold; primary metric MAE |
| `jevbench` | 231 public decisions across all three kinds | Four-/five-level Score; primary public-subset accuracy |
| `open_decisions` | 2,352 decisions across all three kinds | Three- to six-level Score; primary Capability |
| `banking77` | Banking intent classification over all 77 classes | The service must accept all 77 options |
| `reward_bench` | Select the better of two responses | v1 pairwise preferences; existing sample-weighted aggregation |
| `general_decision` | Your local decisions | One question per record; service must support the options/levels |

The currently inspected vLLM Clef-Flash System One path accepts at most 26 Choice options, so it cannot run complete BANKING77. Check actual limits for hosted services as well. Do not prune candidates to bypass a limit.

[JevBench](https://modelscope.cn/datasets/evalscope/jevbench) is the [Benchmark Heaven public subset](https://github.com/fstandhartinger/jevbench), not another benchmark with the same name. It does not produce the full leaderboard Composite. Its family-macro weights task families equally. [OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions) uses the published `train` split for evaluation only and reports Intelligence, Calibration and Capability using its own official protocol.

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

## 6. Check what happens when requests arrive together

Quality evaluation measures decisions; performance testing measures the wait for a decision and the number completed each second.

Stop quality-evaluation traffic on the same service. Fix the model configuration and inputs. These commands use the active terminal's `DECISION_*` settings for Bailian or local Clef-Flash. Respect hosted quotas and begin with a small run.

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

Open `perf_report.html` in the output directory. Check success rate and failures, then p50/p90/p95/p99 latency and successful requests/s. A p95 of 0.3 seconds means about 95% of successful requests finish within that time. One request is one question here, so successful requests/s equals successful decisions/s.

If concurrency raises latency without improving throughput, requests are probably queueing. Network, request length, caching and measurement location also affect comparisons.

> Image placeholder 4 | Successful throughput and p95 latency versus concurrency for one model, with input lengths and cache/serving settings. Add measured results later.

An invalid decision fails even with HTTP 200. Request latency includes HTTP network, queueing and service execution, rather than model compute alone. These non-streaming decisions do not use TTFT/TPOT/ITL. See the [performance quick start](../user_guides/stress_test/quick_start.md) for additional options.

## 7. Use the same format with TypeSafe Jev

[TypeSafe Jev](https://api.typesafe.ai/redoc) also supports System One. Query `/v1/models` with your account, choose an available model, then change the connection settings.

```bash
export TYPESAFE_API_KEY='replace-with-your-TypeSafe-key'
export DECISION_API_KEY="$TYPESAFE_API_KEY"
export DECISION_API_URL='https://api.typesafe.ai/v1'
export DECISION_MODEL='replace-with-a-model-name-from-/v1/models'
python evaluate.py
```

The script still uses `systemone_api`. Bailian and TypeSafe offer their own models and services; the same API format does not guarantee identical weights, probabilities or limits.

EvalScope also supports AutoTrust JEV's `/v1/decide` using `autotrust_api`. It uses a different request structure; see `examples/decision/serve_autotrust.py`. Its fixed-six-level Score cannot cover complete JevBench/OpenDecisions. Ordinary chat models use `openai_api` and can still run ARC, BANKING77, default-Choice BoolQ and RewardBench for answer correctness. Self-reported confidence in chat does not substitute for option probabilities.

## 8. Customize with ms-swift when needed

[ms-swift](https://swift.readthedocs.io/en/latest/BestPractices/Decision.html) provides decision training for JEV-27B-VL, Cloudflare/clef, OmniJev and generic CausalLMs. Trained models can run through Transformers; a compatible decision-service adapter can then connect them to EvalScope for re-evaluation.

`Cloudflare/clef` and `clef-flash` are different models. Exporting a trained ms-swift checkpoint to vLLM System One requires separate adaptation and validation; this guide does not claim that complete training-to-serving path has been verified. Keep training, calibration and evaluation data separate.

The agent's next step was the starting question. You can now examine the answer, the probability and the wait under load, keeping evidence to revisit when the model, rules or training change.

## Troubleshooting

| Symptom | Check first |
|---|---|
| `401` / `403` or unavailable model | API key, region, workspace and model access |
| `404` / `501` | `/v1/systemone` path and a vLLM build with the model's decision strategy |
| Too many Choice options | Actual service limit; BANKING77 requires all 77 options |
| Unsupported Score levels | Preserve the original rubric and use a compatible service |
| Scores with incomplete review counts | Failed/invalid responses, coverage and saved predictions/reviews |
| Want a small first run | Keep `limit=5`; use 20 requests for perf |
