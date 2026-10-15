# Follow-up review of decision evaluation

Reviewed implementation: `2bbbbbad4993594eb92340c8c4d22bcf1b5f08ce`. Results and source/input checksums are in [validation_results.json](validation_results.json); original deployment and quality/perf evidence is in [VALIDATION.md](VALIDATION.md).

## Benchmark audit

| Benchmark | Conversion and scoring checked | Runtime evidence | Ordinary chat |
|---|---|---|---|
| ARC | Choice text order and letter/digit answer mapping; all 1,172 Challenge and 2,376 Easy test label orders match positional labels | Complete Clef Challenge; AutoTrust smoke; fresh qwen-plus smoke | Supported |
| BANKING77 | All 77 numeric labels match intent names across 3,076 rows; no pruning; macro-F1 uses intent identity even after option shuffling | Complete Clef and AutoTrust, all 77 classes; fresh chat smoke | Supported |
| BoolQ | Passage retained; true → Yes, false → No; default Choice and explicit Noul; Noul gold follows meaning after option shuffle | Complete Clef Choice and Noul; AutoTrust Noul smoke; fresh chat smoke | Default Choice only |
| RewardBench v1 | Chosen/rejected positions shuffled deterministically; targets follow positions; sample-weighted subset summaries preserved | Complete Clef filtered split; AutoTrust subset smoke; fresh chat smoke | Supported |
| STS-B | Normalized source gold multiplied by five; continuous labels preserved; expected-score MAE and cross-sample correlations, no invented discrete gold | Complete Clef; six-level AutoTrust smoke | Not supported by this adapter |
| JevBench public | 231 unique public source tasks, original labels/rubrics; official rounding and Noul tie rules; family denominators include missing items; repeated attempts have separate evaluation IDs | Complete Clef; all original per-item scorer outcomes independently compared | Requires a decision API |
| OpenDecisions | Published train split used only for evaluation; original levels retained; raw probabilities passed to the pinned scorer; separate repeat IDs; official missing-prediction denominators | Complete Clef 2,352; all three official summaries independently compared | Requires a decision API |
| General Decision | Standard local JSONL loader; one question per record; gold/provenance excluded from requests; preserves option IDs and ordered levels | Offline native-pipeline/cache/report checks and all three contract types | Requires a decision API |

Ordinary chat smoke used the real OpenAI-compatible qwen-plus endpoint, two examples per benchmark (one RewardBench subset). These are feasibility checks, not complete chat-model quality evaluations. Decision-only adapters reject incompatible chat runs before loading data. Self-reported chat confidence is not converted to candidate probabilities.

## API audit

- `systemone_api`: `/v1/systemone`, one annotated question; exact option/legend validation; scalar Noul is retained as P(yes); raw and normalized scoring distributions are separate. Live model evidence covers all three types and variable Score levels. Permissible JevBench rounding is not rejected by the transport; OpenDecisions still enforces its stricter tolerance.
- `autotrust_api`: `/v1/decide`, aligned arrays mapped to the original option IDs; validates type/options/index/argmax; Score keeps its six levels and rubric instructions. Real JEV evidence covers all three types and 77-way classification; unsupported Score levels are explicitly rejected.
- `openai_api`: unchanged standard chat transport for the four compatible benchmarks. Native probability metrics require a decision response and are not fabricated from text.
- Decision perf plugins reuse complete JSON request bodies and existing HTTP scheduling/storage; no generation defaults or hidden retries. Invalid HTTP 200 responses fail; usage and timeout accounting are checked. All 24,000 saved measured responses pass the updated contracts.

## Reproduced defects and fixes

1. A shuffled BoolQ `Yes` at B was incorrectly interpreted as false in Noul scoring. Gold now follows the selected option's meaning.
2. BANKING77 macro-F1 grouped shuffled positions instead of banking intents. It now uses stable class names, keeping original response probabilities intact.
3. Repeated source tasks collided in JevBench/OpenDecisions official aggregation. Standard per-attempt sample IDs now match tasks and predictions; source IDs remain in provenance.
4. Identical confidence values produced different AURC after reversing sample order. Curves now admit all samples at the same threshold together; the weighted threshold integral is order invariant.
5. Transport rejected a valid JevBench probability sum of 1.015, despite its official 0.02 band. The shared contract accepts that band, while benchmark-specific validation remains stricter where required.
6. A wide vector containing two probabilities of one was accepted as rounding error. The rounding allowance now requires a feasible probability interval around the rounded values.

Regression cases reproduce these failures and pass after the fixes. All 20,962 quality predictions were re-scored; primary scores, official comparisons and coverage are unchanged. One hundred real risk curves and their AURC summaries were independently recomputed.

## Documentation and validation boundary

Both beginner guides now explain which benchmark/API to use, vLLM deployment, a small evaluation before full runs, metric interpretation, local data, and a reproducible perf sweep. All Python configurations, JSONL examples and relative links were checked; both Sphinx builds passed. Platform setup and low-level audit details are excluded from the guides.

The JEV deployment command uses vLLM's official development-wheel installation: source inspection confirms the older 0.20.1 release lacks the text registration, while current main contains the required registration/hybrid-cache/output-head support. CUDA commands have not been executed on NVIDIA hardware in this task. Real backend evidence remains the original validation record; no CUDA or Clef-vLLM runtime claim is made.
