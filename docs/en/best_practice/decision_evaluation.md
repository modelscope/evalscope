# Evaluate decision models with EvalScope

Follow this workflow: **choose a benchmark → serve the model → check quality → measure performance**. Start with a few samples, then remove the limit for a complete evaluation. Run these commands from the EvalScope source repository. Models and datasets are downloaded from ModelScope.

## 1. Choose the model interface and benchmarks

A **chat model** generates an answer through `/v1/chat/completions`; use `eval_type='openai_api'`. A **decision model** returns probabilities through `/v1/decide` or `/v1/systemone`; use `autotrust_api` or `systemone_api`, respectively.

Decision tasks have three types: **Choice** selects one option, **Noul** predicts the probability that a statement is true, and **Score** returns probabilities over ordered levels. For Score, EvalScope calculates an expected rating from that distribution.

| Benchmark | Task | Ordinary chat | System One | AutoTrust JEV example |
|---|---|---|---|---|
| `arc` | Science questions; select ARC-Challenge | Yes | Yes | Yes |
| `banking77` | Banking intent; all 77 options | Yes | Yes | Yes |
| `boolq` | Answer a yes/no question from a passage | Choice | Choice / Noul | Choice / Noul |
| `reward_bench` | Select the better of two answers | Yes | Yes | Yes |
| `stsb` | Sentence similarity, 0–5 | No, for this adapter | Yes | Six-level Score |
| `jevbench` | 231 public decisions of all three types | No | Must support original levels | Cannot run the complete set: four-/five-level Score |
| `open_decisions` | 2,352 decisions of all three types | No | Must support original levels | Cannot run the complete set: three- to six-level Score |
| `general_decision` | Your local decision data | No | Depends on task/levels | Score must have six levels |

Chat support means answer accuracy can be evaluated. Probability metrics require a decision service that returns the complete distribution. A chat model's self-reported confidence is not its token probability.

[JevBench](https://modelscope.cn/datasets/evalscope/jevbench) here is the [Benchmark Heaven public subset](https://github.com/fstandhartinger/jevbench), not another benchmark with the same name. It does not produce the full leaderboard Composite. [OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions) uses its published `train` split for evaluation only; no model training occurs.

## 2. Deploy with vLLM

These commands target Linux with NVIDIA GPUs. Create an isolated environment; in each new terminal, enter the same repository and run `source .venv/bin/activate`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[perf]'
pip install -U uv modelscope fastapi uvicorn httpx
```

Choose one route. Skip to evaluation if you already have a compatible service.

### Route A: an ordinary chat model

Replace the example ModelScope ID with your own, or use a model directory you already downloaded:

```bash
uv pip install -U vllm --torch-backend auto
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ./models/my-chat-model
vllm serve ./models/my-chat-model \
  --served-model-name my-chat-model --port 8000 --max-model-len 8192
```

Keep that terminal running. Check it from another terminal:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-chat-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

Use the same served model name in EvalScope.

### Route B: AutoTrust decisions

The following uses `autotrust/JEV-27B`; it needs substantially more GPU memory than the small chat example. Use vLLM's multi-GPU configuration if necessary. This model uses a recent architecture, so install a development build using the [official vLLM instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). The download script fetches and checks the complete release:

```bash
uv pip install -U vllm --torch-backend auto \
  --extra-index-url https://wheels.vllm.ai/nightly
python examples/decision/download_models.py \
  --model autotrust/JEV-27B --output ./models/JEV-27B
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

The response should contain `options` and aligned `probabilities`. Your own decision model must provide a compatible `/v1/decide` or `/v1/systemone` service. This wrapper expects the AutoTrust release format; changing the model name alone does not adapt a different model. Standard vLLM chat serving does not automatically provide these decision routes.

## 3. Run a small evaluation

Save the appropriate example as `evaluate.py`, then run `python evaluate.py`.

### Ordinary chat: four existing benchmarks

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='my-chat-model',
    eval_type='openai_api',
    api_url='http://127.0.0.1:8000/v1',
    api_key='EMPTY',
    datasets=['arc', 'banking77', 'boolq', 'reward_bench'],
    dataset_args={'arc': {'subset_list': ['ARC-Challenge']}},
    generation_config={'temperature': 0, 'max_tokens': 256},
    limit=5,
))
```

BoolQ uses its default Choice mode. Do not enable Noul or select benchmarks marked as incompatible with chat.

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

Stop quality-evaluation traffic on the same service first. The following uses the AutoTrust decision service; ordinary chat performance testing is covered by the [performance quick start](../user_guides/stress_test/quick_start.md).

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

1. **Success rate, failures and timeouts**: an invalid answer fails even with HTTP 200.
2. **p50 / p90 / p95 / p99 latency**: about 50% / 90% / 95% / 99% of requests finish within this time.
3. **Successful requests/s**: higher means more completed decisions per second; there is one decision per request here.
4. **Concurrency trade-off**: more concurrency may increase queueing without improving throughput.

TTFT/TPOT/ITL do not apply to non-streaming decisions. Request latency includes network and queueing. Compare runs using the same inputs, serving settings and measurement location.

## Troubleshooting

| Symptom | Action |
|---|---|
| HTTP 404 | Check API type and port; a plain chat server does not expose native decision routes |
| Unsupported model or `lm_head` LoRA | Use a compatible vLLM build required by the model and retain its adapter/head |
| Score requires six levels | Use a service supporting the benchmark's original levels; do not change the benchmark |
| New decision benchmark rejects a chat configuration | Follow the compatibility table and change the benchmark or API |
| Want a quick check first | Use `limit=5` for quality and 20 requests for perf |
| Need to inspect an answer | Check its input, original options, prediction and review |
