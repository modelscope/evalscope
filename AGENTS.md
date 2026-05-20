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
make lint                                                                       # required before commit (yapf + isort + flake8 + codespell)
pytest tests/cli/test_all.py::TestRun::test_ci_lite -v -s -p no:warnings        # CI smoke test
pytest tests/perf/test_perf.py::TestPerf::test_run_perf_multi_parallel -v -s    # perf
```

Commits failing `make lint` are rejected on `main`.

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
- **Quotes** governed by `double-quote-string-fixer` hook — follow existing file style; do not mix.
- **f-strings** for formatting (no `%` or `.format()` unless necessary).
- **Imports**: isort with `first_party = evalscope`, groups `STDLIB / THIRDPARTY / LOCALFOLDER`, `multi_line_output=3`.
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

**flake8 ignore list** (`setup.cfg`): `F401, F403, F405, F821, W503, E251, W504, F824, F541, E501, E226, E121-E131`. Do not expand — new ignores must be justified in the PR.

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

**Adapter base classes** (extend, don't reinvent): `DefaultDataAdapter`, `MultiChoiceAdapter`, `VisionLanguageAdapter`, `Text2ImageAdapter`, `ImageEditAdapter`, `NERAdapter`, `AgentAdapter`. Optional capabilities via mixins: `LLMJudgeMixin`, `SandboxMixin`.

**Non-native backends** live under `evalscope/backend/` (OpenCompass, VLMEvalKit, RAGEval) and are dispatched from `run.py` with their own BackendManager.

## Adding a benchmark

1. Create `evalscope/benchmarks/<name>/<name>_adapter.py`.
2. Extend `DefaultDataAdapter`, override `record_to_sample()` (and optionally `sample_to_fewshot()`, `extract_answer()`).
3. Decorate with `@register_benchmark(BenchmarkMeta(name=..., ...))`.
4. Auto-discovered by globbing `evalscope/benchmarks/*/**/*_adapter.py`.
5. Add a smoke test.

## Conventions & gotchas

- `eval_type`: `openai_api`, `llm_ckpt`, `mock_llm`, `text2image`, `image_editing`. Deprecated aliases: `server` → `openai_api`, `checkpoint` → `llm_ckpt`.
- `limit`: `int` = count, `float` = fraction.
- `repeats`: duplicates items for k-metrics. `generation_config.n` is deprecated and mapped.
- Use `generation_config` for runtime params. `TaskConfig.timeout` / `stream` are deprecated — forwarded with a warning.
- `dataset_args` merges into `BenchmarkMeta._update()` (supports `local_path`, `filters` OrderedDict prepended).
- Models are memoized by `(name, config, base_url, api_key, args)`.
- Use `@thread_safe` for model creation, `run_in_threads_with_progress` for concurrent eval.
- Outputs land in `outputs/<timestamp>/{logs,predictions,reviews,reports,configs}/` (see `OutputsStructure`). `use_cache` resumes runs; `rerun_review` recomputes scores only.
- `evalscope/app/` is **deprecated** — use `evalscope service` for the Web dashboard.

## Submission

```bash
make dev      # once
make lint     # before every commit
pytest tests/cli/test_all.py::TestRun::test_ci_lite -v -s -p no:warnings
```
