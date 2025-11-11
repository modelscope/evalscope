# Copilot instructions for this repo (evalscope)

Use these repo-specific rules to ship changes confidently. Prefer concrete patterns from this codebase.

## Architecture and flow
- CLI entry: `evalscope` (see `pyproject.toml -> [project.scripts]`) with subcommands:
  - `eval` → native evaluation (`evalscope/cli/start_eval.py` → `evalscope/run.py`).
  - `perf` → service load testing (extra `evalscope[perf]`).
  - `app` → result visualization (extra `evalscope[app]`).
- Native eval path (`evalscope/run.py`):
  1) Build `TaskConfig` (`evalscope/config.py`) → set `work_dir` → `OutputsStructure` creates `logs/`, `predictions/`, `reviews/`, `reports/`, `configs/`.
  2) Build `Model` via API registry (see `evalscope/api/model/model.py`, function `get_model_with_task_config`).
  3) For each dataset, resolve benchmark via `get_benchmark` (`evalscope/api/registry.py`) → run `DefaultEvaluator` (`evalscope/evaluator/evaluator.py`).
  4) Caching (JSONL) and a JSON report + pretty table (`evalscope/report`).
- Non-native backends (OpenCompass, VLMEvalKit, RAGEval) are dispatched in `run.py`; output dirs are normalized (e.g., set `work_dir/time_str`).

## Conventions and extension points
- Registries (`evalscope/api/registry.py`):
  - Benchmarks: decorate a `DataAdapter` with `@register_benchmark(BenchmarkMeta(...))`. See `evalscope/benchmarks/**`.
  - Model APIs: `@register_model_api(name)` maps `TaskConfig.eval_type` → provider. See `evalscope/models/model_apis.py`.
  - Metrics: `@register_metric(name)`; Aggregators: `@register_aggregation(name)`. See `evalscope/metrics/metric.py`.
  - Filters: `@register_filter(name)` to post-process multiple generations. See `evalscope/filters/selection.py`.
- DataAdapter contract (`evalscope/api/benchmark/benchmark.py`): implement
  `load_dataset`, `run_inference`, `calculate_metrics`, `aggregate_scores`, `generate_report`, `finalize`.
  Configure behavior in `BenchmarkMeta` (`meta.py`): subsets, metrics, prompts, filters, aggregation, tags.
- Model selection: `eval_type` values include `openai_api`, `llm_ckpt`, `mock_llm`, `text2image`, `image_editing` (see deprecations below). Models are memoized by (name, config, base_url, api_key, args).
- Caching & outputs: under `outputs/<timestamp>/...` (see `OutputsStructure`). `use_cache` resumes; `rerun_review` recomputes scores only.

## Developer workflows
- Install for dev: `pip install -e .` (or `make dev` for `dev,perf,docs` + pre-commit). Extras in `pyproject.toml`.
- Quick eval (CLI): `evalscope eval --model Qwen/Qwen2.5-0.5B-Instruct --datasets gsm8k --limit 5`.
- Quick eval (Python): `from evalscope import run_task, TaskConfig; run_task(TaskConfig(model='Qwen/Qwen2.5-0.5B-Instruct', datasets=['gsm8k'], limit=5))`.
- API eval (OpenAI-compatible): set `--eval-type openai_api` or `TaskConfig(eval_type='openai_api')`, provide `--api-url/--api-key`.
- Perf benchmark: `evalscope perf --url http://127.0.0.1:8801/v1/chat/completions --api openai --model qwen2.5 --parallel 5 -n 20` (see `test_perf.sh`).
- Visualization: `evalscope app` (see README “Visualization of Evaluation Results”).

## Behavior that matters
- Concurrency: predictions use `eval_batch_size`; reviews use `judge_worker_num` (thread pools in `DefaultEvaluator`).
- `limit`: int (count) or float (fraction). `repeats` duplicates items for k-metrics; `generation_config.n` is deprecated and mapped.
- `dataset_args` merges into `BenchmarkMeta._update()` (supports `local_path`, `filters` OrderedDict prepended).
- Reports: printed tables via `evalscope/report`; failures are logged but should not crash runs.
- Cache JSONL schemas: see `evalscope/api/evaluator/cache.py` (`ModelResult`, `ReviewResult`).

## Deprecations / gotchas
- Prefer `openai_api` and `llm_ckpt`; `server` and `checkpoint` are deprecated aliases (`models/model_apis.py`).
- Use `generation_config` for runtime params; `TaskConfig.timeout/stream` warn and are forwarded.

## Where to look
- Orchestration: `evalscope/run.py`, `evalscope/cli/*`.
- API layer: `evalscope/api/**` (model/messages/tool/dataset/metric/filter/benchmark/registry).
- Evaluator: `evalscope/evaluator/evaluator.py` (caching, parallelism, reporting).
- Extensions: `evalscope/benchmarks/**`, `evalscope/metrics/metric.py`, `evalscope/filters/**`, `evalscope/models/**`.

## Code Quality

- Always use English comments
- Type hints required for all code
- Public APIs must have docstrings
- Functions must be focused and small
- Follow existing patterns exactly
- Class names in PascalCase
- Constants in UPPER_SNAKE_CASE
- Document with docstrings
- Use f-strings for formatting

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint


## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale
