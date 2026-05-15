# AGENTS.md

EvalScope contributor map: architecture index, tech stack, and enforced coding standards. This document is a **map**, not a manual — drill into the referenced modules for full detail.

---

## 1. Project Architecture

EvalScope is an LLM evaluation framework built on a **registry-based plugin architecture**. Core flow: `TaskConfig → run_task() → Model + DataAdapter → Evaluator → Report`.

### 1.1 Top-level layout

| Path | Responsibility |
| --- | --- |
| `evalscope/api/` | Core abstractions: `benchmark/`, `model/`, `metric/`, `filter/`, `dataset/`, `evaluator/`, `registry.py` |
| `evalscope/benchmarks/` | Benchmark implementations (one `*_adapter.py` per subdir, auto-discovered) |
| `evalscope/models/` | Model API providers: `openai_api`, `llm_ckpt`, `mock_llm`, `text2image`, ... |
| `evalscope/evaluator/` | `DefaultEvaluator`: orchestrates inference, caching, scoring, aggregation |
| `evalscope/metrics/` | Built-in metrics & aggregators; `t2v_metrics/` for vision metrics |
| `evalscope/filters/` | Post-processing filters on model outputs |
| `evalscope/backend/` | Non-native backends: `opencompass`, `vlm_eval_kit`, `rag_eval` (dispatched in `run.py`) |
| `evalscope/perf/` | HTTP load testing for model services |
| `evalscope/service/` | Flask REST API + Web Dashboard (frontend under `web/`) |
| `evalscope/agent/` | Agent execution engine (AgentLoop, SWE-bench protocol, ...) |
| `evalscope/collections/` | Multi-benchmark sampled evaluation |
| `evalscope/sandbox/` | Sandboxed code execution (Docker / ms-enclave) |
| `evalscope/cli/`, `run.py`, `config.py` | CLI entry, dispatch, `TaskConfig` |

### 1.2 Core abstractions (`evalscope/api/`)

- **`DataAdapter`** — benchmark contract: `load_dataset` / `run_inference` / `calculate_metrics` / `aggregate_scores` / `generate_report` / `finalize`. Common variants: `DefaultDataAdapter`, `MultiChoiceAdapter`, `VisionLanguageAdapter`, `Text2ImageAdapter`, `AgentAdapter`.
- **`ModelAPI` / `Model`** — generation layer; memoized by `(name, config, base_url, api_key, args)`.
- **`Metric` / `Filter` / `Aggregation`** — scoring & post-processing primitives.
- **`Sample`** (Pydantic): `input`, `target`, `choices`, `metadata`, `tools`, `sandbox`, `files`.
- **Mixins**: `LLMJudgeMixin`, `SandboxMixin` mixed into adapters as needed.

### 1.3 Registry extension points (`evalscope/api/registry.py`)

```python
@register_benchmark(BenchmarkMeta(name=..., ...))   # DataAdapter
@register_model_api(name)                            # ModelAPI provider
@register_metric(name) / @register_aggregation(name) # metrics & aggregators
@register_filter(name)                               # output filters
@register_evaluator(name)                            # evaluators
```

Adding a benchmark: create `evalscope/benchmarks/<name>/<name>_adapter.py` → extend `DefaultDataAdapter` → implement `record_to_sample()` (and optionally `sample_to_fewshot()`, `extract_answer()`) → decorate with `@register_benchmark`. Auto-discovered by globbing `*_adapter.py`.

### 1.4 Outputs & cache

`outputs/<timestamp>/` contains `logs/`, `predictions/`, `reviews/`, `reports/`, `configs/` (see `OutputsStructure`). `use_cache` resumes runs; `rerun_review` recomputes scores only. Cache schema: `evalscope/api/evaluator/cache.py`.

---

## 2. Tech Stack

### 2.1 Language & runtime

- **Python** ≥ 3.10 (3.10 / 3.11 / 3.12 supported)
- **Build**: `setuptools>=69` + `wheel`, configured in `pyproject.toml`
- **License**: Apache-2.0

### 2.2 Core runtime dependencies (`requirements/framework.txt`)

| Category | Libraries |
| --- | --- |
| Model / inference | `openai`, `litellm>=1.55,<2.0`, `transformers>=4.33,!=4.57.2`, `modelscope[datasets]>=1.34` |
| Data validation | **`pydantic`** (mandatory for typed data models), `jsonschema` |
| Data processing | `pandas`, `pillow`, `jsonlines`, `pyyaml>=5.1`, `more_itertools` |
| Eval / math | `sympy`, `latex2sympy2_extended`, `pylatexenc`, `word2number`, `editdistance`, `sacrebleu`, `rouge-score>=0.1.0`, `rouge-chinese`, `nltk`, `jieba`, `zhconv` |
| Tooling / UI | `rich`, `tqdm`, `tabulate`, `plotly`, `colorlog`, `Markdown`, `jinja2` |
| Misc | `requests`, `overrides`, `docstring_parser`, `dotenv` |

### 2.3 Optional extras (`pyproject.toml`)

`opencompass` / `vlmeval` / `rag` / `perf` / `app` / `aigc` / `sandbox` / `service` / `dev` / `docs` / `all`, plus per-benchmark extras (`swe_bench`, `bfcl`, `ifeval`, ...).

### 2.4 Toolchain

- **Code style**: `yapf` v0.43, `isort` v7.0, `flake8` v7.3, `codespell`
- **Pre-commit hooks**: `trailing-whitespace`, `end-of-file-fixer`, `double-quote-string-fixer`, `check-yaml`, `mixed-line-ending=lf` (see `.pre-commit-config.yaml`)
- **Tests**: `pytest` (config in `pyproject.toml [tool.pytest.ini_options]`; `testpaths=tests`, `python_files=*test*.py`)
- **CLI entry point**: `evalscope = evalscope.cli.cli:run_cmd`
- **Frontend** (Web Dashboard): `evalscope/web/` (Node project; build artifacts under `evalscope/web/dist/**`)

---

## 3. Coding Standards (Enforced)

> All rules below are enforced via `make lint` (pre-commit). Local pass is required before submitting.

### 3.1 Formatting

- **Line width 120** (yapf / isort / flake8 aligned).
- **4-space indent**, no tabs.
- **LF line endings**; single trailing newline at EOF.
- **Imports** via isort: `first_party = evalscope`, groups `STDLIB / THIRDPARTY / LOCALFOLDER`, `multi_line_output=3` with trailing comma.
- **Quotes**: governed by the `double-quote-string-fixer` hook — follow the existing file style; do not mix.
- **Strings**: prefer **f-strings**; avoid `%` formatting and `.format()` unless necessary.

### 3.2 Naming

| Element | Style | Example |
| --- | --- | --- |
| Class | `PascalCase` | `DefaultDataAdapter` |
| Function / method / variable | `snake_case` | `run_inference` |
| Constant | `UPPER_SNAKE_CASE` | `DEFAULT_TIMEOUT` |
| Private | leading `_` | `_internal_state` |
| Handler function | `handle_` prefix | `handle_request` |
| Module file | `snake_case.py`; benchmark adapters end with `_adapter.py` | `gsm8k_adapter.py` |

### 3.3 Types & comments

- **Type hints required** on all function signatures (params + return).
- **English only** for comments and docstrings.
- **Public APIs must have docstrings**; internal helpers only when intent is non-obvious.
- Mark pending work with `# TODO:`.

### 3.4 Structure & design

- **Early returns** to avoid nested conditionals.
- **DRY**, but do not over-abstract just to remove minor duplication.
- **Minimal changes**: only touch code related to the current task; no drive-by cleanup.
- **Small, focused functions**; define composing functions before their components.
- **Use `Arguments` / `TaskConfig`** (Pydantic) for configuration — never raw dicts at module boundaries.
- **Pydantic-first**: cross-module data contracts must use Pydantic models.
- **Reuse existing patterns**: new benchmarks / models / metrics must go through the existing registries and adapter base classes — no parallel mechanisms.

### 3.5 Lint ignore policy (`setup.cfg`)

flake8 currently ignores: `F401, F403, F405, F821, W503, E251, W504, F824, F541, E501, E226, E121-E131`. **Do not expand the ignore list**; new ignores must be justified in the PR.

### 3.6 Tests

- Tests live in `tests/`; files match `*test*.py`, classes `Test*`, functions `test_*`.
- New benchmarks / models / metrics **must** ship a minimal runnable test (see `tests/cli/test_all.py::TestRun::test_ci_lite`).
- Mock external services in tests; no reliance on real network / paid APIs.

### 3.7 Submission workflow

```bash
make dev      # install dev extras + pre-commit
make lint     # required before commit
pytest tests/cli/test_all.py::TestRun::test_ci_lite -v -s -p no:warnings  # CI-equivalent smoke test
```

Commits failing `make lint` are not accepted on the main branch.

---

## 4. Quick Index

| I want to... | Go to |
| --- | --- |
| Add a benchmark | `evalscope/benchmarks/<existing>/` + `api/benchmark/benchmark.py` |
| Add a Model API | `evalscope/models/model_apis.py` + `api/model/model.py` |
| Add a metric / filter | `evalscope/metrics/metric.py` / `evalscope/filters/selection.py` |
| Read the main flow | `evalscope/run.py` → `evalscope/evaluator/evaluator.py` |
| Inspect config schema | `evalscope/config.py` (`TaskConfig`) |
| Read the CLI | `evalscope/cli/` |
| Read the registry | `evalscope/api/registry.py` |
| Detailed rules | `CLAUDE.md`, `.github/copilot-instructions.md`, `setup.cfg`, `.pre-commit-config.yaml` |
