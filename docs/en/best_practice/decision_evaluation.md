# Evaluating decision models

Use EvalScope to retain and evaluate the complete probability distribution of a **Choice**, **Noul** (probability of yes), or **Score** (ordered levels) decision. Correct answers, probability quality, execution coverage, and serving performance are separate measurements.

The Chinese walkthrough contains the complete deployment and troubleshooting procedure. All example commands run from the EvalScope repository root.

## Benchmarks and protocols

| Benchmark | Evaluation data | Main metric / boundary |
|---|---|---|
| `arc` | `allenai/ai2_arc`, ARC-Challenge/test | Accuracy |
| `banking77` | `mteb/banking77`, test | Accuracy; all 77 labels retained; macro-F1 |
| `boolq` | `google/boolq`, validation | Accuracy; Choice by default, explicit Noul mode |
| `reward_bench` | `allenai/reward-bench`, filtered | v1 pairwise accuracy; existing sample-weighted aggregation |
| `stsb` | `sentence-transformers/stsb`, test | Expected-value MAE; continuous source gold multiplied by 5 |
| `jevbench` | `evalscope/jevbench`, test | Public-subset accuracy, family macro, per-kind metrics |
| `open_decisions` | `evalscope/OpenDecisions`, train | Official Capability; evaluation only |
| `general_decision` | Local `default_test.jsonl` | Accuracy; one question per record |

`systemone_api` accepts a base URL ending in `/v1` and calls `/systemone`. `autotrust_api` calls `/decide`. The latter maps the returned probability array back to the original option IDs without reordering. Both preserve raw responses and validated distributions. A chat-completions endpoint alone is insufficient: load the decision adapter/head and apply its released readout and calibration.

The ModelScope JevBench mirror contains Benchmark Heaven's 231 public tasks (72 original, 48 easy, 111 hard), pinned to `609a664e81f46e03731600bd7bd9f1b7a61257c5`. It is distinct from the similarly named coherence benchmark. The mirror includes original JSONL, lossless Parquet, source licenses and SHA-256 hashes. Public results are not the full leaderboard Composite/Capability.

## Download and serve

```bash
pip install -e '.[perf]'
pip install modelscope transformers accelerate safetensors fastapi uvicorn httpx
python examples/decision/download_models.py \
  --model Cloudflare/clef-flash --output ./models/clef-flash
python examples/decision/download_models.py \
  --model autotrust/JEV-27B --output ./models/JEV-27B
CUDA_VISIBLE_DEVICES=0 python examples/decision/serve_clef.py \
  --model-path ./models/clef-flash --port 18131
```

The downloader verifies the named ModelScope ref against the pinned commit's file tree, then verifies every file. Some SDK versions reject commit IDs in the snapshot validator; the hash checks preserve the content pin while using a named ref for transfer. All models and evaluation data are downloaded from ModelScope.

Clef uses the released PyTorch implementation, including the joint schema head. This reference service serializes requests and is not a vLLM batching benchmark.

AutoTrust uses a vLLM completions engine plus a thin `/decide` service:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ./models/JEV-27B \
  --served-model-name autotrust/JEV-27B \
  --enable-lora --max-lora-rank 32 \
  --lora-modules jev-decision=./models/JEV-27B/adapter_vllm \
  --max-model-len 16384 --max-logprobs 256 \
  --logprobs-mode processed_logprobs --port 18132
python examples/decision/serve_autotrust.py \
  --model-path ./models/JEV-27B --upstream http://127.0.0.1:18132/v1 --port 18133
```

This requires support for Qwen3.5 hybrid attention/GDN and `lm_head` LoRA. The PPU vendor build requires initializing `/usr/local/PPU_SDK/envsetup.sh` and installing the isolated registration plugin under `examples/decision/ppu_vllm`. See the Chinese walkthrough for exact commands. Keep the matched vendor torch/vLLM environment; generic CUDA wheels are not a PPU upgrade path.

The AutoTrust wrapper follows the released bare prompt, verbalizer bias and per-kind temperature, with thinking off. It rejects missing/nonfinite candidate logprobs. Choice above the native 16 training slots uses the released single-pass extension, retaining every option. Score has exactly six levels (0–5); other scales fail explicitly. Clef supports the original variable-level rubrics needed for complete JevBench and OpenDecisions runs.

## Evaluate

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='Cloudflare/clef-flash', eval_type='systemone_api',
    api_url='http://127.0.0.1:18131/v1',
    datasets=['jevbench', 'open_decisions', 'stsb'],
    eval_batch_size=4,
    generation_config={'timeout': 300, 'retries': 1},
    limit=5,  # remove for full evaluation; applies per subset
))
```

For P0, use `datasets=['arc','banking77','boolq','reward_bench','stsb']` and `dataset_args={'arc': {'subset_list': ['ARC-Challenge']}}`. To select BoolQ Noul, set `dataset_args={'boolq': {'extra_params': {'decision_kind': 'noul', 'decision_threshold': 0.5}}}`. AutoTrust uses `eval_type='autotrust_api'` and base URL `http://127.0.0.1:18133/v1`.

Gold answers, rationale, teacher predictions and provenance are never sent to the model. Do not set generation controls such as temperature, max_tokens, tools or streaming. `retries=1` means one total attempt.

A local `default_test.jsonl` accepts:

```json
{"id":"c1","state":"I am happy.","question":{"type":"choice","instructions":"Classify sentiment.","criteria":{"positive":"Positive","negative":"Negative"}},"expected":"positive"}
{"id":"n1","state":"The sky is blue.","question":{"type":"noul","instructions":"Is the sky blue?"},"expected":"yes"}
{"id":"s1","state":"Excellent service.","question":{"type":"score","instructions":"Rate satisfaction.","criteria":["Poor","Fair","Good","Excellent"]},"expected":"3"}
```

Select `general_decision` and `dataset_args={'general_decision': {'local_path': './decision-data'}}`. For fixed-category classification, set `fixed_classes=true` on records to enable macro-F1. Question-local A/B/C/D positions are not global classes.

## Interpret probability metrics

Each report declares one primary metric. JevBench public accuracy and family macro include missing predictions as wrong. Its pinned public scorer selects no for a Noul tie, retains probabilities within a 0.001 sum tolerance, and normalizes only above that within 0.02. Ordinary valid-sample metrics are separate. NLL uses an explicit `1e-12` floor. Binary Brier is `(p_yes-y)^2`; multiclass Brier sums class-wise squared errors. They are not averaged together. AUPRC uses the average-precision step integral with grouped ties. ECE uses ten equal-width bins; binary ECE uses P(yes), categorical ECE uses top-label confidence. Permitted rounding normalization is recorded separately from unchanged raw probabilities.

Different Score level counts are reported separately, keeping MAE and correlations on a common scale within each group. Score retains its expected level and ordered distribution. For example, probabilities `[0.1,0.2,0.3,0.4]` have expectation 2.0 and argmax 3. Continuous STS-B gold is not discretized for nRPS or accuracy. Spearman, Pearson, AUROC and AUPRC are computed across samples; undefined values are omitted with reasons and effective counts. Risk is classification error for Choice/Noul and normalized expected-value absolute error for Score, including continuous gold. Risk–coverage curves/AURC distinguish top-option probability from provider confidence; ties use stable sample order.

Ordinary metrics use valid responses; inspect execution coverage. OpenDecisions additionally follows its pinned MIT scorer (`5026a13ad6e5a8606810a3a5c4e410d8d6638e71`), using all selected tasks as the official denominator, including missing predictions as wrong in Intelligence. Its Noul threshold is 0.5, unlike some JevBench abstention rules. Limited runs are not full leaderboard submissions. Official 0–100 values are stored as 0–1 ratios and displayed as percentages.

Export the persisted curves to standalone interactive HTML:

```bash
python examples/decision/plot_reliability.py \
  --report outputs/<run>/reports/<model>/jevbench.json --output reliability.html
```

## Benchmark serving performance

Run perf without other evaluation traffic on the same service. Export separate fixed request sets:

```bash
python examples/decision/prepare_perf_requests.py \
  --api systemone --model Cloudflare/clef-flash --output ./perf-input/systemone
for kind in choice noul score; do
  evalscope perf --api systemone --model Cloudflare/clef-flash \
    --url http://127.0.0.1:18131/v1 --dataset line_by_line \
    --dataset-path "./perf-input/systemone/$kind.jsonl" \
    --parallel 1 4 8 16 --number 1000 1000 1000 1000 \
    --warmup-num 20 --no-stream --read-timeout 300 \
    --outputs-dir "./perf-results/clef-$kind"
done
```

For JEV, export `--api autotrust --model autotrust/JEV-27B`, then select `--api autotrust`, its input directory and port 18133. The plugins inject no generation parameters and perform no hidden retries. An invalid HTTP 200 decision is a failed request.

Read summary/percentile JSON and HTML for p50/p90/p95/p99 latency, successful requests/s, failures, Timeout Requests and input throughput. Each concurrency level excludes 20 warmups and measures 1,000 requests. One request contains one decision, so successful requests/s equals decisions/s. Input throughput divides successful input tokens by the measurement window. TTFT/TPOT/ITL are explicitly listed as not applicable; compatibility zero fields do not mean zero latency.

Archive request manifests, actual usage lengths, model/code revisions, versions, cache settings and device occupancy. HTTP latency includes network, service queueing, the full response and JSON decoding. Request preparation and protocol validation are outside that timer; the throughput window includes client processing. Server-internal timing is separate. Measure localhost on the PPU server rather than through an SSH tunnel. Higher concurrency can primarily increase queueing in the serial Clef reference service.

The actual validation platform is PPU-ZW810. Clef's PyTorch path and AutoTrust's vLLM path must be reported separately. Ordinary CUDA commands are deployment examples, not evidence of CUDA validation. Final quality/perf evidence accompanies the change's validation record.
