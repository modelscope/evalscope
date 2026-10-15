# Decision evaluation validation — 2026-10-15

This record distinguishes real PPU inference, cached prediction verification, local tests, and CI. Machine-readable results, request checksums, database checksums, exact model revisions, and per-run coverage are in [validation_results.json](validation_results.json). Full model file hashes are in [model_revisions.json](model_revisions.json).

## Platform and provenance

- Initial implementation review/replay: `6e9970881fdf8a3cf28de6ba0e893db78513e00c`. Follow-up review/replay and re-scoring: `2bbbbbad4993594eb92340c8c4d22bcf1b5f08ce`; serving examples and model releases are unchanged.
- Python 3.12.14, torch `2.10.0+v0.1.0.ppu2.1.0.oe`, vLLM `0.20.1+v0.1.0.ppu2.1.0.oe`, transformers 5.8.1, ModelScope 1.36.3, datasets 4.8.4. PPU SDK driver 1.5.0-299c95; HGGC 13.0.
- One PPU-ZW810 (96 GiB) per model, isolated devices 7 / 6. Clef uses its official PyTorch joint head, eager attention and a serial service. JEV uses the released LoRA/head/bias/calibration through vendor vLLM and the bundled architecture-registration plugin, eager execution, 16,384 context, prefix caching disabled, memory utilization 0.85. No shared environment upgrade or source patch.
- ModelScope releases: Clef `0dc1ab273435d4681bce5996f50b8d79fb79bb83`, JEV `4163b1f068332b00cbdab60938a3e84e260f2498`. All 17 / 65 release files were SHA-256 verified, including inference code and calibration.
- Inference and performance data were collected during development. Every saved wire request/response was replayed against the archived commit (20,962 evaluation predictions plus 24,000 measured perf responses); every quality prediction was re-scored. Both committed serving examples were then restarted and passed six fresh live requests with immutable model revision assertions. An unsupported four-level AutoTrust Score was explicitly rejected before HTTP.
- All model and dataset downloads on PPU used ModelScope. JevBench mirror has 231 unique IDs and four verified configurations; its raw files pin upstream `609a664e81f46e03731600bd7bd9f1b7a61257c5`. Data revision/checksum details remain separate from model revisions.

## Real quality evaluation

All listed requests completed with valid distributions and matching review counts; errors = 0. Values are ratios except STS-B MAE (0–5 units). ARC is Challenge/test; BoolQ is validation; RewardBench is the existing v1 filtered split; BANKING77 is the actual MTEB test split (3,076 rows, all 77 classes). AutoTrust rows other than BANKING77 are explicitly smoke runs; RewardBench smoke selects two samples from each of 23 subsets.

| Model / task | Requested = reviewed | Primary metric | Value |
|---|---:|---|---:|
| clef/arc | 1,172 | accuracy | 0.9855 |
| clef/banking77 | 3,076 | accuracy | 0.9581 |
| clef/boolq | 3,270 | accuracy | 0.9122 |
| clef/boolq-noul | 3,270 | accuracy | 0.9098 |
| clef/jevbench | 231 | public_accuracy | 0.8225 |
| clef/open_decisions | 2,352 | capability | 0.6720 |
| clef/reward_bench | 2,985 | accuracy | 0.8435 |
| clef/stsb | 1,379 | mae | 0.6484 |
| jev/arc | 5 | accuracy | 1.0000 |
| jev/banking77 | 3,076 | accuracy | 0.7724 |
| jev/boolq | 50 | accuracy | 0.8200 |
| jev/reward_bench | 46 | accuracy | 0.8478 |
| jev/stsb | 50 | mae | 0.6055 |

Clef JevBench family-macro = 0.8270. Its 231 per-task validity/correctness and ordinal expectations were checked against the pinned upstream scorer. OpenDecisions' independent original `score.py` returned Intelligence 55.3260%, Calibration 79.0701%, Capability 67.1981%; all three agree with EvalScope within report rounding. Its scorer revision is `5026a13ad6e5a8606810a3a5c4e410d8d6638e71`, SHA-256 `06994fe9fffca7819c34b3dd985ab4fdcdd00fa115d8e37a98076377dcb0d03e`. NLL, Brier, score expectation and MAE were independently recomputed from real review rows. Both BANKING77 runs retained every option and all 77 gold classes. No model-training contamination audit was performed; these numbers validate integration, not a new leaderboard claim.

## Real performance sweep

Each model/kind uses a fixed set of 20 gold-free requests, repeats it at concurrency 1/4/8/16, warms up 20 times per setting and measures 1,000 requests per setting. Choice uses full 77-way BANKING77; Noul uses BoolQ; Score uses six-level STS-B. Exact UTF-8 lengths, hashes, observed token lengths, duration, input throughput and database SHA-256 are in the JSON record. All 24 configurations have zero failures and zero timeouts. Every saved SQLite response was independently parsed and its usage checked against the final contracts.

HTTP latency includes localhost transport, queueing, inference, full response reception and JSON decoding. Preparation and protocol validation occur outside the HTTP timer; throughput uses the full measurement window. No server-only duration was emitted. TTFT/TPOT/ITL are not applicable. Requests contain one decision, so successful requests/s equals decisions/s. There is no hidden retry or generated text workload. These two releases differ in architecture, request rendering and backend; this is a reproducibility check, not a controlled model speed ranking.

| Model | Kind | Concurrent | Success / total | req/s | p50 s | p90 s | p95 s | p99 s | Mean input tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clef | choice | 1 | 1000 / 1000 | 3.8880 | 0.2555 | 0.2651 | 0.2678 | 0.2909 | 1651.50 |
| clef | choice | 4 | 1000 / 1000 | 3.9475 | 1.0007 | 1.0470 | 1.0531 | 1.0765 | 1651.50 |
| clef | choice | 8 | 1000 / 1000 | 3.8431 | 2.0448 | 2.1452 | 2.1613 | 2.1882 | 1651.50 |
| clef | choice | 16 | 1000 / 1000 | 4.0865 | 3.8681 | 3.8953 | 3.9074 | 3.9250 | 1651.50 |
| clef | noul | 1 | 1000 / 1000 | 7.3115 | 0.1357 | 0.1430 | 0.1448 | 0.1561 | 298.40 |
| clef | noul | 4 | 1000 / 1000 | 7.3268 | 0.5429 | 0.5546 | 0.5592 | 0.5711 | 298.40 |
| clef | noul | 8 | 1000 / 1000 | 7.2826 | 1.0890 | 1.1067 | 1.1131 | 1.1743 | 298.40 |
| clef | noul | 16 | 1000 / 1000 | 7.2343 | 2.1758 | 2.2070 | 2.2197 | 2.2724 | 298.40 |
| clef | score | 1 | 1000 / 1000 | 7.5106 | 0.1324 | 0.1360 | 0.1373 | 0.1433 | 225.45 |
| clef | score | 4 | 1000 / 1000 | 7.5318 | 0.5278 | 0.5365 | 0.5396 | 0.5599 | 225.45 |
| clef | score | 8 | 1000 / 1000 | 7.4899 | 1.0582 | 1.0755 | 1.0814 | 1.1099 | 225.45 |
| clef | score | 16 | 1000 / 1000 | 7.4787 | 2.1024 | 2.1277 | 2.1637 | 2.2058 | 225.45 |
| jev | choice | 1 | 1000 / 1000 | 4.6512 | 0.2124 | 0.2228 | 0.2327 | 0.2640 | 538.50 |
| jev | choice | 4 | 1000 / 1000 | 6.8955 | 0.5759 | 0.5961 | 0.6032 | 0.6262 | 538.50 |
| jev | choice | 8 | 1000 / 1000 | 7.4728 | 1.0630 | 1.0740 | 1.0787 | 1.1004 | 538.50 |
| jev | choice | 16 | 1000 / 1000 | 8.0234 | 1.9840 | 1.9955 | 1.9983 | 2.0065 | 538.50 |
| jev | noul | 1 | 1000 / 1000 | 4.7186 | 0.2090 | 0.2197 | 0.2338 | 0.2658 | 187.40 |
| jev | noul | 4 | 1000 / 1000 | 8.7543 | 0.4523 | 0.4761 | 0.4858 | 0.5039 | 187.40 |
| jev | noul | 8 | 1000 / 1000 | 12.9922 | 0.6176 | 0.6487 | 0.6549 | 0.6799 | 187.40 |
| jev | noul | 16 | 1000 / 1000 | 14.7392 | 1.0876 | 1.1085 | 1.1122 | 1.1250 | 187.40 |
| jev | score | 1 | 1000 / 1000 | 4.7369 | 0.2090 | 0.2190 | 0.2267 | 0.2485 | 101.45 |
| jev | score | 4 | 1000 / 1000 | 8.7503 | 0.4528 | 0.4728 | 0.4845 | 0.5190 | 101.45 |
| jev | score | 8 | 1000 / 1000 | 14.6245 | 0.5404 | 0.5572 | 0.5615 | 0.5928 | 101.45 |
| jev | score | 16 | 1000 / 1000 | 19.3471 | 0.8200 | 0.8320 | 0.8369 | 0.8732 | 101.45 |

Clef's throughput stays nearly flat as concurrency increases because its reference service is serial; queueing raises latency. JEV's batched vLLM path increases throughput. No measured performance threshold was imposed.

## Local validation and limits

- After merging current main: 1,164 targeted/model/perf/registry/report/evaluator tests passed; 47 dashboard/report API regression tests also passed. Offline tests use local HTTP fixtures. CI-lite separately passed a live qwen-plus smoke.
- `make lint`, `make lint-imports` (5 contracts), generated frontend contract checks/build and Chinese Sphinx HTML build passed. GitHub workflow status is reported on the PR and is not substituted by these local results.
- JSON/HTML reports, cache-only re-review, official missing-prediction denominators, unavailable metrics and reliability curves were checked. Runtime artifacts remain in the isolated `decision-evaluation` directory on the validation host: `clef-eval/`, `jev-eval/`, `perf/`, verification logs, release manifests and `cleanup-evidence.json`. No credentials are committed.
- AutoTrust supports six-level Score only; complete JevBench/OpenDecisions use Clef. The 77-choice path uses the upstream single-pass extension beyond its 16 trained choice slots. Ordinary CUDA deployment commands are examples, not CUDA test evidence. Clef's dedicated head was not validated under vLLM.
- Clef's released encoder has a 16,384-token state window. The maximum observed input usage across these complete runs was 4,032 tokens (JevBench); no selected input hit the window. This does not validate longer inputs.
- Only this task's services and engine were stopped. Ports 18131/18132/18133 are closed, no owned process remains, and devices 6/7 returned to 10 MiB with 0% utilization and no running processes. Logs and result artifacts were retained.

## Follow-up review

The follow-up reproduced and fixed choice-shuffle semantics (BoolQ Noul and BANKING77 macro-F1), repeat-ID collisions in both official aggregators, tied-confidence AURC ordering, and probability-rounding boundary checks. Typed decision adapters now reject chat configurations before downloading data. The beginner guides were rewritten around vLLM, API/benchmark selection, results and perf; platform-specific setup and low-level validation details remain outside those guides.

After these fixes, **1,175 targeted tests**, CI-lite, lint/import contracts and both language documentation builds passed. Real qwen-plus chat smoke completed **two samples each** on ARC-Challenge, BANKING77, BoolQ and RewardBench (alpacaeval-easy), with all eight predictions and reviews valid. A source audit checked all ARC-Challenge 1,172 / ARC-Easy 2,376 choice-label orders and all BANKING77 3,076 numeric/text labels; no mismatches were found.

All **20,962** saved quality predictions were re-scored; the independent pinned official scorers and all primary scores/counts remain unchanged. All quality wire records and **24,000** perf wire records were replayed with the follow-up implementation. An independent threshold-group calculation checked **100** real risk-coverage curves and AURC summaries. This phase reused real saved inference; it does not claim a new complete inference run or CUDA runtime validation.

The ordinary CUDA guide follows upstream vLLM installation and native AutoTrust serving requirements. The v0.20.1 source lacks the text architecture registration; the current main implementation has registration, hybrid-cache support and output-head LoRA mapping. The guide therefore uses the official development-wheel installation path for JEV. This is source verification, separate from the real vendor-vLLM runtime evidence above. Chat support remains limited to the four compatible benchmarks; the new typed adapters do not invent probability distributions from generated answers.
