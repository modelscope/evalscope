# AGENTS.md

EvalScope — LLM evaluation framework with a registry-based plugin architecture. This file is the contract for AI coding agents working in this repo.

## Setup

```bash
pip install -e .       # basic install
make dev               # dev + perf + docs extras + pre-commit
```

Python ≥ 3.10 (3.10 / 3.11 / 3.12). Dependencies: `requirements/framework.txt` + `pyproject.toml [project.optional-dependencies]` (extras: `opencompass`, `vlmeval`, `rag`, `perf`, `app`, `aigc`, `sandbox`, `service`, `dev`, `docs`, `all`, plus per-benchmark extras).

## Build, lint, test

```bash
make format                                                                     # apply Ruff lint fixes, import sorting, and formatting
make lint                                                                       # required before commit (Ruff + basic pre-commit hooks)
pytest tests/cli/test_all.py::TestRun::test_ci_lite -v -s -p no:warnings        # CI smoke test
pytest tests/perf/test_perf_basic.py::TestPerfBasic::test_multi_parallel_sweep -v -s    # perf
```

Commits failing `make lint` are rejected on `main`.

## Docs generation

Benchmark detail pages (`docs/{zh,en}/benchmarks/<name>.md`) and meta cache (`evalscope/benchmarks/_meta/<name>.json`) are **auto-generated** from each adapter's `BenchmarkMeta.description` + dataset statistics. Do not hand-edit those files.

Every `BenchmarkMeta.description` must be English Markdown with these sections in this order:

1. `## Overview`: benchmark purpose and scope.
2. `## Task Description`: bullet fields for `Task Type`, `Input`, `Output`, and `Domain` (use a more precise fourth field such as `Modalities` or `Grading` only when `Domain` does not apply).
3. `## Key Features`: dataset scale/source, evaluated capabilities, and version-specific behavior.
4. `## Evaluation Notes`: metrics, scoring procedure, runtime/dependency requirements, and compatibility limits.

Do not replace these required headings with benchmark-specific headings. Add extra sections only when the four required sections are insufficient.

When you add a benchmark or change its `BenchmarkMeta.description`, run:

```bash
make docs-pipeline BENCHMARK="<name1> <name2>" FORCE=1   # update _meta JSON + translate descriptions to zh
make docs-generate                                        # render .md files from _meta
```

Targets: `docs-update` (meta only), `docs-update-stats` (+ dataset statistics), `docs-translate` (zh), `docs-pipeline` (stats + translate), `docs-generate` (.md), `docs` (full Sphinx HTML build).

Conventions:
- `BENCHMARK="a b c"` selects benchmarks; omit for `--all`.
- `FORCE=1` appends `--force` to recompute even if data is cached.
- `WORKERS=N` parallelism (default 4).
- `--translate` calls an LLM; needs `DASHSCOPE_API_KEY` (or equivalent) in env.

## Quick eval

```bash
evalscope eval --model Qwen/Qwen2.5-0.5B-Instruct --datasets gsm8k --limit 5
```

```python
from evalscope import run_task, TaskConfig
run_task(TaskConfig(model='Qwen/Qwen2.5-0.5B-Instruct', datasets=['gsm8k'], limit=5))
```

## Code style (enforced)

- **Line width 120**, 4-space indent, LF endings, trailing newline at EOF.
- **Quotes**: single quotes, enforced by the Ruff formatter.
- **Linting**: Ruff's `E`, `F`, and `W` rules for maintained source files.
- **Imports**: Ruff's `I` rules, with `evalscope` detected as first-party and standard import sections.
- **f-strings** for formatting (no `%` or `.format()` unless necessary).
- **Type hints required** on every function signature.
- **English only** for comments and docstrings.
- **Public APIs need docstrings**; internal helpers only when intent is non-obvious.
- `# TODO:` prefix for pending work.

| Element | Style |
| --- | --- |
| Class | `PascalCase` |
| Function / variable | `snake_case` |
| Constant | `UPPER_SNAKE_CASE` |
| Private | `_leading_underscore` |
| Handler function | `handle_` prefix |
| Benchmark adapter file | `<name>_adapter.py` |

**Ruff ignore list** (`pyproject.toml`): `E501, E741, F401, F403, F405, F541, F821`. Do not expand — new ignores must be justified in the PR.

## Design rules

- **Early returns** over nested conditionals.
- **Minimal changes**: only touch code related to the current task; no drive-by cleanup.
- **Pydantic-first**: cross-module data contracts use Pydantic models. Use `TaskConfig` / `Arguments` for configuration — never raw dicts at module boundaries.
- **Reuse existing patterns**: new benchmarks / models / metrics go through existing registries and adapter base classes — no parallel mechanisms.
- **DRY** but don't over-abstract just to remove minor duplication.

## Tests

- Live under `tests/`; files `*test*.py`, classes `Test*`, functions `test_*`.
- New benchmark / model / metric **must** ship a minimal runnable test (pattern: `tests/cli/test_all.py::TestRun::test_ci_lite`).
- Mock external services — no reliance on real network / paid APIs.

## Architecture pointers

Don't try to learn the architecture from this file — read these and grep:

| Topic | Source of truth |
| --- | --- |
| Main flow | `evalscope/run.py` → `evalscope/evaluator/evaluator.py` |
| Config schema | `evalscope/config.py` (`TaskConfig`) |
| Registries | `evalscope/api/registry.py` |
| Benchmark contract | `evalscope/api/benchmark/benchmark.py` (`DataAdapter`, `BenchmarkMeta`) |
| Model layer | `evalscope/api/model/model.py`, `evalscope/models/model_apis.py` |
| CLI dispatch | `evalscope/cli/` |
| Cache schema | `evalscope/api/evaluator/cache.py` |

**Registry decorators**: `@register_benchmark`, `@register_model_api`, `@register_metric`, `@register_aggregation`, `@register_filter`, `@register_evaluator`.

**Adapter base classes** (extend, don't reinvent): `DefaultDataAdapter`, `MultiChoiceAdapter`, `VisionLanguageAdapter`, `Text2ImageAdapter`, `ImageEditAdapter`, `NERAdapter`, `AgentAdapter`. Optional capabilities via mixins: `LLMJudgeMixin`, `CodeExecutionSandboxMixin`.

**Non-native backends** live under `evalscope/backend/` (OpenCompass, VLMEvalKit, RAGEval) and are dispatched from `run.py` with their own BackendManager.

## Adding a benchmark

1. Create `evalscope/benchmarks/<name>/<name>_adapter.py`.
2. Extend `DefaultDataAdapter`, override `record_to_sample()` (and optionally `sample_to_fewshot()`, `extract_answer()`).
3. Reuse the standard dataset flow (`load_subset()` and existing `DataLoader` implementations) for shuffle, limit, repeats, filtering, conversion, and indexing. Override the full `load()` flow only when the standard loaders cannot represent the source format, and keep custom loading limited to benchmark-specific parsing or validation.
4. Use `download_dataset_file()` or `download_dataset_snapshot()` for benchmark media and raw files; do not duplicate hub resolution, cache, path-safety, or download state inside an adapter.
5. Decorate with `@register_benchmark(BenchmarkMeta(name=..., ...))`.
6. Auto-discovered by globbing `evalscope/benchmarks/*/**/*_adapter.py`.
7. Add a smoke test.

### Evaluation versioning

`BenchmarkMeta.evaluation_version` is the published version of a benchmark's evaluation semantics. New benchmarks
must declare their initial version explicitly. Raise the minor version when data, sample conversion, default prompt,
choice/target mapping, default scoring/judge, or aggregation semantics change; raise the major version for a rename
or task-definition replacement. Documentation, tests, and pure refactors do not change it.

## Adding a judge-scored benchmark

An adapter must **never** call `self.llm_judge.judge()` or parse a judge reply itself — that debt is fenced off by `tests/api/judge/test_gates.py`, which scans every file under `evalscope/benchmarks/` (helpers included, so moving a parser into `utils.py` does not evade it). Score through the JSON output contract in `evalscope/api/judge/` instead:

1. Pick a `scoring_policy` (`JUDGE_ONLY` or `JUDGE_DEFAULT`). Judge scoring always goes through the contract; there is no opt-in flag and no legacy path.
2. **Single verdict per sample:** implement `judge_definition(context)` and return `JudgeDefinition.labels(...)` for a label mapping or `JudgeDefinition.numeric(...)` for a 0-1 rating. A generic `prompt_template` must state grading criteria only: `OutputContract.instruction()` appends the reply format. An adapter that preserves an official fixed output template may keep that format instruction instead, provided its `OutputContract` schema matches the official template.
3. **Custom shape (multiple cases, ratings, rubrics):** `judge_definition(context)` declares a Pydantic `schema_model`, wraps it in `OutputContract`, and returns `JudgeDefinition.workflow(...)`:
   - `cases` contains one `JudgeCase(case_id, output_contract, metadata)` per thing to judge.
   - `request(case, placement, completed, context)` renders messages and appends `case.output_contract.instruction()` so the prompt and parser cannot drift. An official fixed output template may be used instead when its required fields and constraints match the case's `OutputContract` schema.
   - `reduce(verdicts, context)` folds parsed verdicts into `{metric: value}`. Read a verdict's context from `CaseVerdict.metadata`, never by parsing `case_id`.
   - Optional `expand`, `fallback`, and `finalize` callbacks handle staged cases, rule fallbacks, and score finalization. They may be nested functions or private adapter helpers, but are passed only through the returned definition.
4. **Rule short-circuit:** if deterministic scoring settles the sample before judge I/O, return `JudgeDefinition.skip(score, reason='...')`. The non-empty reason is persisted in `Score.metadata` as `judge_skipped=True` and `judge_skip_reason`; the web review panel displays it as rule-based scoring.
5. The executor owns request execution, position swap, repeats, multi-judge aggregation and fail-closed exclusion. Transport retries belong to the model implementation; a reply that fails the contract is not automatically retried and excludes the sample from the metric — never scored 0 or full credit — so a metric's `num` can be below the sample count.
6. Add a scripted-judge test in `tests/api/judge/test_migrated_adapters.py` covering: a valid verdict, a parse failure (prose / malformed), and a transport `[ERROR]` — each must exclude, not silently score. A judge double must carry the surface the definition reads (`score_type`, `score_mapping`, `build_prompt`), and be injected through the `llm_judge` setter rather than a private attribute.

## Conventions & gotchas

- `eval_type`: `openai_api`, `llm_ckpt`, `mock_llm`, `text2image`, `image_editing`. Deprecated aliases: `server` → `openai_api`, `checkpoint` → `llm_ckpt`.
- `limit`: `int` = count, `float` = fraction.
- `repeats`: duplicates items for k-metrics. `generation_config.n` is deprecated and mapped.
- Use `generation_config` for runtime params. `TaskConfig.timeout` / `stream` are deprecated — forwarded with a warning.
- `dataset_args` merges into `BenchmarkMeta._update()` (supports `local_path`, `filters` OrderedDict prepended).
- Models are memoized by `(name, config, base_url, api_key, args)`.
- Use `@thread_safe` for model creation, `run_in_threads_with_progress` for concurrent eval.
- Outputs land in `outputs/<timestamp>/{logs,predictions,reviews,reports,configs}/` (see `OutputsStructure`). `use_cache` resumes runs; `rerun_review` recomputes scores only.
- `evalscope app` CLI command is **deprecated** (see `evalscope/cli/start_app.py`) — use `evalscope service` for the Web dashboard.

## Submission

```bash
make dev      # once
make format   # apply safe Ruff fixes and formatting
make lint     # before every commit
pytest tests/cli/test_all.py::TestRun::test_ci_lite -v -s -p no:warnings
```
