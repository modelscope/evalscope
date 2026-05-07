# Contributing to EvalScope

Thank you for considering contributing to EvalScope! This guide covers everything you need to get started.

## Table of Contents

- [Contributing to EvalScope](#contributing-to-evalscope)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Development Setup](#development-setup)
    - [Backend (Python)](#backend-python)
    - [Frontend (React + Vite)](#frontend-react--vite)
    - [Full-Stack Development](#full-stack-development)
  - [Project Structure](#project-structure)
  - [Adding a New Benchmark](#adding-a-new-benchmark)
    - [Step 1: Create the adapter directory](#step-1-create-the-adapter-directory)
    - [Step 2: Write the adapter](#step-2-write-the-adapter)
    - [Key methods you can override](#key-methods-you-can-override)
    - [BenchmarkMeta key fields](#benchmarkmeta-key-fields)
    - [Step 3: Add extra dependencies (if any)](#step-3-add-extra-dependencies-if-any)
    - [Step 4: Update documentation (optional)](#step-4-update-documentation-optional)
    - [Verify your benchmark](#verify-your-benchmark)
  - [Code Quality](#code-quality)
    - [Linting](#linting)
    - [Testing](#testing)
  - [Git Workflow](#git-workflow)

---

## Quick Start

```bash
# 1. Fork & clone
git clone https://github.com/<your-username>/evalscope.git
cd evalscope

# 2. Install in editable mode with dev dependencies
make dev

# 3. Install pre-commit hooks
pre-commit install
```

---

## Development Setup

### Backend (Python)

EvalScope requires **Python >= 3.10**.

```bash
# Base install (editable)
pip install -e .

# With all dev tools
pip install -e '.[dev,perf,docs]'

# With the web service
pip install -e '.[service]'
```

**Run the backend service:**

```bash
evalscope service --host 0.0.0.0 --port 9000
```

Optional dependency groups (install via `pip install -e '.[<group>]'`):

| Group | Purpose | Key Packages |
|-------|---------|-------------|
| `dev` | Testing & linting | pytest, pytest-cov |
| `service` | Web dashboard & REST API | flask, plotly |
| `perf` | Performance benchmarking | — |
| `docs` | Documentation build | sphinx |
| `rag` | RAG evaluation | — |
| `aigc` | AIGC evaluation | — |
| `sandbox` | Sandboxed code execution | ms-enclave |

Some benchmarks have their own extra dependencies (e.g. `pip install -e '.[bfcl]'`).

### Frontend (React + Vite)

The dashboard is a React SPA located at `evalscope/web/`.

```bash
# Install dependencies
make web-install

# Start dev server (hot reload, proxies API to localhost:9000)
make web-dev

# Production build
make web-build
```

The dev server runs at `http://localhost:5173` and automatically proxies `/api/v1/*` and `/health` to the backend at `http://127.0.0.1:9000`.

**Tech stack:** React 19 · TypeScript · Vite · Tailwind CSS 4 · React Router · Plotly.js

### Full-Stack Development

For the best development experience, run both servers simultaneously:

```bash
# Terminal 1: Backend
evalscope service --debug

# Terminal 2: Frontend (hot reload)
make web-dev
```

Open `http://localhost:5173` in your browser — changes to frontend code are reflected instantly.

---

## Project Structure

```
evalscope/
├── api/              # Core API: registry, benchmark base classes, dataset, metric, model
│   ├── benchmark/    #   DataAdapter, BenchmarkMeta, adapter subclasses
│   ├── dataset/      #   Dataset loading, Sample dataclass
│   ├── evaluator/    #   TaskState, evaluation loop
│   ├── messages/     #   Chat message types
│   ├── metric/       #   Score, AggScore, metric registry
│   ├── model/        #   Model abstraction (OpenAI-compatible)
│   └── registry.py   #   register_benchmark(), BENCHMARK_REGISTRY
│
├── benchmarks/       # All benchmark adapters (auto-discovered)
│   └── <name>/
│       ├── __init__.py
│       └── <name>_adapter.py
│
├── cli/              # CLI entry points (evalscope eval/perf/service/app)
├── constants.py      # Global constants & tags
├── perf/             # Performance benchmarking subsystem
├── report/           # Report generation & visualization
├── service/          # Flask REST API + SPA serving
│   ├── app.py        #   Flask app factory, run_service()
│   └── blueprints/   #   API route handlers (eval, perf, reports)
├── utils/            # Shared utilities (logging, IO, etc.)
└── web/              # React SPA (dashboard UI)
    ├── src/
    │   ├── api/      #   API client & type definitions
    │   ├── components/  # UI components
    │   ├── pages/    #   Route pages
    │   └── i18n/     #   Internationalization
    └── vite.config.ts
```

---

## Adding a New Benchmark

EvalScope uses a **decorator-based registry** pattern. Adding a benchmark requires only two files.

### Step 1: Create the adapter directory

```
evalscope/benchmarks/my_benchmark/
├── __init__.py          # (empty)
└── my_benchmark_adapter.py
```

Adapters are **auto-discovered**: any `*_adapter.py` under `evalscope/benchmarks/` is automatically imported at startup, which triggers the `@register_benchmark` decorator.

### Step 2: Write the adapter

Choose a base class depending on your benchmark type:

| Base Class | Use When |
|------------|----------|
| `DefaultDataAdapter` | General text QA, math, coding |
| `MultiChoiceAdapter` | Multiple-choice questions |
| `AgentAdapter` | Function calling, tool use |
| `VisionLanguageAdapter` | Image + text (VQA, etc.) |
| `MultiTurnAdapter` | Multi-turn conversations |
| `Text2ImageAdapter` | Text-to-image generation |
| `NERAdapter` | Named entity recognition |
| `ImageEditAdapter` | Image editing |

**Minimal example** (text QA benchmark):

```python
from typing import Any, Dict
from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

DESCRIPTION = """
## Overview
Brief description of what this benchmark evaluates.

## Task Description
- **Task Type**: ...
- **Input**: ...
- **Output**: ...

## Evaluation Notes
- Default configuration uses **0-shot** evaluation
"""

@register_benchmark(
    BenchmarkMeta(
        name='my_benchmark',           # unique identifier (snake_case)
        pretty_name='MyBenchmark',      # display name
        dataset_id='org/dataset-name',  # ModelScope / HuggingFace dataset ID
        tags=[Tags.REASONING],          # category tags
        description=DESCRIPTION,
        subset_list=['default'],        # dataset subsets
        metric_list=['acc'],            # evaluation metrics
        eval_split='test',              # split to evaluate on
        few_shot_num=0,                 # number of few-shot examples
        prompt_template='{question}',   # prompt template with placeholders
    )
)
class MyBenchmarkAdapter(DefaultDataAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a dataset row to a Sample object."""
        return Sample(
            input=record['question'],
            target=record['answer'],
        )
```

### Key methods you can override

| Method | Purpose | When to Override |
|--------|---------|------------------|
| `record_to_sample()` | Map dataset row → `Sample` | Always |
| `extract_answer()` | Extract structured answer from model output | When default extraction is insufficient |
| `match_score()` | Custom scoring logic | When `acc` / built-in metrics don't fit |
| `sample_to_fewshot()` | Format a sample as few-shot example | When using few-shot with custom format |
| `_on_inference()` | Custom model interaction | For agent/tool-use benchmarks |

### BenchmarkMeta key fields

```python
BenchmarkMeta(
    name='...',              # Required: unique snake_case ID
    dataset_id='...',        # Required: remote dataset ID or local path
    pretty_name='...',       # Display name
    tags=[...],              # From evalscope.constants.Tags
    description='...',       # Markdown description for docs
    subset_list=['default'], # Dataset subsets
    metric_list=['acc'],     # Metric names or dicts: [{'acc': {'numeric': True}}]
    aggregation='mean',      # 'mean', 'pass@k', 'f1', etc.
    eval_split='test',       # Evaluation split name
    train_split='train',     # Training split (for few-shot)
    few_shot_num=0,          # Few-shot count
    prompt_template='...',   # Prompt template with {placeholders}
    filters=OrderedDict(),   # Output filters
    extra_params={},         # Additional configurable parameters
    sandbox_config={},       # Sandboxed execution config (for code benchmarks)
    review_timeout=None,     # Per-sample timeout in seconds
)
```

### Step 3: Add extra dependencies (if any)

If your benchmark needs additional packages, create a `requirements.txt` in the benchmark directory:

```
evalscope/benchmarks/my_benchmark/requirements.txt
```

Then register it in `pyproject.toml`:

```toml
[tool.setuptools.dynamic.optional-dependencies]
my_benchmark = {file = ["evalscope/benchmarks/my_benchmark/requirements.txt"]}
```

Users can install via `pip install 'evalscope[my_benchmark]'`.

### Step 4: Update documentation (optional)

Run the doc pipeline to auto-generate benchmark documentation:

```bash
make docs-update BENCHMARK=my_benchmark
make docs-translate BENCHMARK=my_benchmark
make docs-generate
```

### Verify your benchmark

```bash
# Check it's registered
evalscope eval --benchmarks my_benchmark --model dummy --limit 5

# Run via service
evalscope service
# Then use the Web dashboard or API: POST /api/v1/eval/invoke
```

---

## Code Quality

### Linting

This project uses **pre-commit** with the following hooks:

- **flake8** — Python style checker
- **isort** — Import sorting
- **yapf** — Code formatting
- Trailing whitespace, YAML checks, line ending fixes

```bash
# Run all checks
make lint
# or
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
pytest tests/

# Run a specific test
pytest tests/benchmark/test_xxx.py
```

---

## Git Workflow

1. **Create a branch** with a descriptive name:
   ```bash
   git checkout -b feature/my-benchmark
   ```

2. **Make your changes** and commit with clear messages:
   ```bash
   git commit -m "feat: add MyBenchmark adapter"
   ```

3. **Run quality checks** before pushing:
   ```bash
   pre-commit run --all-files
   pytest tests/
   ```

4. **Push and open a Pull Request** against the `main` branch. Provide a clear description of your changes.

---

Thank you for your contribution!
