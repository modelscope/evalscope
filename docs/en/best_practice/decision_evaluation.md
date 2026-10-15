# From correct answers to trustworthy probabilities: evaluate decision models with EvalScope

Follow this workflow: **choose a benchmark → serve the model → check quality → measure performance**. Start with a few samples, then remove the limit for a complete evaluation. Run these commands from the EvalScope source repository. Models and datasets are downloaded from ModelScope.

## Why this evaluation matters

Two decision models can both answer 80% of questions correctly while being very different decision makers: one may assign 51% probability to its mistakes, while another assigns 99%. Accuracy cannot show that difference; probability metrics such as NLL, Brier and ECE can.

This guide is only for decision models that **natively return complete probability distributions**:

- If you already have a compatible `/v1/decide` or `/v1/systemone` service, skip deployment and go to section 3.
- To deploy the AutoTrust JEV example, prepare the model and decision endpoint in section 2.
- If local resources cannot host the example, connect to a compatible remote decision service. Do not substitute post-processed confidence for native model probabilities.

For a first read, complete sections 1, 3 and 4. Use section 2 only when you need to deploy a service, and sections 5–6 for custom data or performance testing.

## 1. Choose the decision interface and benchmarks

A decision model returns option probabilities, a truth probability or an ordered score distribution. Use `eval_type='autotrust_api'` for `/v1/decide`, or `eval_type='systemone_api'` for `/v1/systemone`.

Decision tasks have three types: **Choice** selects one option, **Noul** predicts the probability that a statement is true, and **Score** returns probabilities over ordered levels. For Score, EvalScope calculates an expected rating from that distribution.

| Benchmark | Task | System One | AutoTrust JEV example |
|---|---|---|---|
| `arc` | Science questions; select ARC-Challenge | Yes | Yes |
| `banking77` | Banking intent; all 77 options | Yes | Yes |
| `boolq` | Answer a yes/no question from a passage | Choice / Noul | Choice / Noul |
| `reward_bench` | Select the better of two answers | Yes | Yes |
| `stsb` | Sentence similarity, 0–5 | Yes | Six-level Score |
| `jevbench` | 231 public decisions of all three types | Must support original levels | Cannot run the complete set: four-/five-level Score |
| `open_decisions` | 2,352 decisions of all three types | Must support original levels | Cannot run the complete set: three- to six-level Score |
| `general_decision` | Your local decision data | Depends on task/levels | Score must have six levels |

EvalScope can calculate the probability metrics in this guide only when the service returns the complete distribution. [JevBench](https://modelscope.cn/datasets/evalscope/jevbench) here is the [Benchmark Heaven public subset](https://github.com/fstandhartinger/jevbench), not another benchmark with the same name. It does not produce the full leaderboard Composite. [OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions) uses its published `train` split for evaluation only; no model training occurs.

## 2. Deploy with vLLM

These commands target Linux with NVIDIA GPUs. Create an isolated environment; in each new terminal, enter the same repository and run `source .venv/bin/activate`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[perf]'
pip install -U uv modelscope fastapi uvicorn httpx
```

Skip to evaluation if you already have a compatible decision service.

### AutoTrust JEV decision model

The following uses `autotrust/JEV-27B`; it needs substantial GPU memory. Use vLLM's multi-GPU configuration if necessary. This model uses a recent architecture, so install a development build using the [official vLLM instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), then download the complete model from ModelScope:

```bash
uv pip install -U vllm --torch-backend auto \
  --extra-index-url https://wheels.vllm.ai/nightly
modelscope download --model autotrust/JEV-27B \
  --local_dir ./models/JEV-27B
vllm serve ./models/JEV-27B \
  --served-model-name autotrust/JEV-27B \
  --enable-lora --max-lora-rank 32 \
  --lora-modules jev-decision=./models/JEV-27B/adapter_vllm \
  --max-model-len 16384 --max-logprobs 256 \
  --logprobs-mode processed_logprobs --port 18132
```

In another terminal, expose the decision interface:

```bash
python examples/decision/serve_autotrust.py \
  --model-path ./models/JEV-27B \
  --upstream http://127.0.0.1:18132/v1 --port 18133
```

**vLLM runs on 18132; EvalScope calls the decision service on 18133.** Retain the decision adapter/head and use a vLLM build supporting the architecture and output-head LoRA, as required by the [model's deployment instructions](https://modelscope.cn/models/autotrust/JEV-27B).

```bash
curl http://127.0.0.1:18133/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"kind":"choice","state":"The sky is blue.","question":"What color is the sky?","options":["Blue","Red"],"thinking":"off"}'
```

The response should contain `options` and aligned `probabilities`. For a formal evaluation, record the downloaded model revision, vLLM version and serving arguments so that a moving model alias does not make the run irreproducible. Your own decision model must provide a compatible `/v1/decide` or `/v1/systemone` service. This wrapper expects the AutoTrust release format; changing the model name alone does not adapt a different model. A standard vLLM OpenAI-compatible endpoint does not automatically provide these decision routes.

## 3. Run a small evaluation

Save the appropriate example as `evaluate.py`, then run `python evaluate.py`.

### AutoTrust: Choice, Noul and Score

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='autotrust/JEV-27B',
    eval_type='autotrust_api',
    api_url='http://127.0.0.1:18133/v1',
    datasets=['arc', 'banking77', 'boolq', 'reward_bench', 'stsb'],
    dataset_args={
        'arc': {'subset_list': ['ARC-Challenge']},
        'boolq': {'extra_params': {'decision_kind': 'noul'}},
    },
    eval_batch_size=4,
    generation_config={'timeout': 300, 'retries': 1},
    limit=5,
))
```

ARC, BANKING77 and RewardBench test Choice; BoolQ tests Noul; STS-B tests six-level Score. Decision requests do not use temperature, max_tokens or streaming. `retries=1` means one total attempt per sample.

### System One: complete JevBench / OpenDecisions

Use a service that supports the tasks' original Score levels:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-decision-model',
    eval_type='systemone_api',
    api_url='http://127.0.0.1:9000/v1',
    datasets=['jevbench', 'open_decisions'],
    eval_batch_size=4,
    generation_config={'timeout': 300, 'retries': 1},
    limit=5,
))
```

Replace the model name and URL with your actual System One service. Do not remap four-/five-level tasks to AutoTrust's six-level scale.

**Remove `limit=5` after the smoke run to evaluate the complete split.** Limits apply per subset; a benchmark with multiple subsets can issue more than five requests. Datasets load automatically from ModelScope. Gold answers are not sent to the model.

## 4. Read the results

The terminal prints the report directory, usually `outputs/<timestamp>/reports/`. Open `report.html` for results; inspect `predictions/` and `reviews/` for individual samples. First check successful requests and valid review counts, then interpret scores.

| Metric | Meaning | Better direction |
|---|---|---|
| Accuracy | Fraction of correct answers | Higher |
| macro-F1 | Give each class equal importance | Higher |
| Precision / Recall / F1 | Accuracy of positive Noul answers / positive cases found / balance between the two | Higher |
| NLL / Brier | Reward useful probabilities; penalize confident mistakes | Lower |
| ECE | Whether stated confidence matches observed correctness | Lower; also check accuracy |
| AUROC / AUPRC | Separate positive and negative Noul examples | Higher; compare datasets cautiously |
| MAE | Average rating error | Lower |
| Spearman / Pearson | Agreement in ranking / score variation | Closer to 1 |
| nRPS | Put Score probability near the correct level | Lower |
| AURC | Error risk when answering higher-confidence questions first | Lower |

Score MAE uses the expected rating; discrete Score accuracy uses the most probable level. On STS-B's 0–5 scale, MAE 0.6 means an average error of about 0.6 points. Continuous gold is preserved, so this adapter does not report classification accuracy or nRPS for STS-B. Some metrics need enough distinct observations; missing values are not zero.

JevBench's primary metric is public-subset accuracy, with family-macro also reported. OpenDecisions uses Capability and also reports Intelligence (task ability) and Calibration (probability reliability). These have their own scoring rules. A ratio of `0.82` in JSON means 82%; MAE stays in rating units. Limited runs are not complete leaderboard results. RewardBench uses v1 pairwise decisions and a sample-weighted summary, rather than the official category-weighted leaderboard score.

## 5. Evaluate your own data

Create `decision-data/default_test.jsonl`, one question per line:

```json
{"id":"c1","state":"The customer cannot find their card.","question":{"type":"choice","instructions":"What should support help with?","criteria":{"find":"Find the card","close":"Close the account"}},"expected":"find"}
{"id":"n1","state":"The sky is blue.","question":{"type":"noul","instructions":"Is the sky blue?"},"expected":"yes"}
{"id":"s1","state":"Excellent service.","question":{"type":"score","instructions":"Rate satisfaction.","criteria":["Very poor","Poor","Fair","Good","Very good","Excellent"]},"expected":"5"}
```

`state` is the input material, `question` defines the task, and `expected` is the gold answer used only for scoring. Gold is an option ID for Choice, `yes` / `no` for Noul, or a zero-based level number for Score.

In your working decision configuration replace:

```python
datasets=['general_decision'],
dataset_args={'general_decision': {'local_path': './decision-data'}},
```

Keep the other model/API settings. AutoTrust requires six Score levels; other services must support the levels you specify.

## 6. Measure performance

Stop quality-evaluation traffic on the same service first. The following uses the AutoTrust decision service; general performance-testing concepts are covered by the [performance quick start](../user_guides/stress_test/quick_start.md).

Export fixed requests: full 77-way BANKING77 Choice, BoolQ Noul and STS-B Score.

```bash
python examples/decision/prepare_perf_requests.py \
  --api autotrust --model autotrust/JEV-27B --output ./perf-input/autotrust
for kind in choice noul score; do
  evalscope perf --api autotrust --model autotrust/JEV-27B \
    --url http://127.0.0.1:18133/v1 --dataset line_by_line \
    --dataset-path "./perf-input/autotrust/$kind.jsonl" \
    --parallel 1 4 8 16 --number 1000 1000 1000 1000 \
    --warmup-num 20 --no-stream --read-timeout 300 \
    --outputs-dir "./perf-results/$kind"
done
```

Each kind runs at concurrency 1/4/8/16, with 20 warmups followed by 1,000 measured requests per setting. For an initial check, use `--parallel 1 --number 20 --warmup-num 2`. To test System One, use `--api systemone` in both export and perf commands and change the model, URL and input directory. The files are specific to their protocol.

Open `perf_report.html` in the output directory and read:

1. **Success rate and failures**: an invalid answer fails even with HTTP 200.
2. **p50 / p90 / p95 / p99 latency**: about 50% / 90% / 95% / 99% of requests finish within this time.
3. **Successful requests/s**: higher means more completed decisions per second; there is one decision per request here.
4. **Concurrency trade-off**: more concurrency may increase queueing without improving throughput.

TTFT/TPOT/ITL do not apply to non-streaming decisions. Request latency includes network and queueing. Compare runs using the same inputs, serving settings and measurement location.

## Troubleshooting

| Symptom | Action |
|---|---|
| HTTP 404 | Check API type and port; a plain model endpoint does not expose native decision routes |
| Unsupported model or `lm_head` LoRA | Use a compatible vLLM build required by the model and retain its adapter/head |
| Score requires six levels | Use a service supporting the benchmark's original levels; do not change the benchmark |
| A decision benchmark rejects the API configuration | Follow the compatibility table and select the matching decision API |
| Want a quick check first | Use `limit=5` for quality and 20 requests for perf |
| Need to inspect an answer | Check its input, original options, prediction and review |
