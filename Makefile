# default rule
default: install

# ============================================================================
# Documentation Generation
# ============================================================================
#
# WORKFLOW (full pipeline):
#   docs-update → docs-translate → docs-generate → docs-en / docs-zh
#
# STEP-BY-STEP:
#   Step 1  docs-update[/stats]  Read adapter metadata → write _meta/<name>.json
#   Step 2  docs-translate       Translate readme.en → readme.zh via LLM API
#   Step 3  docs-generate        Read all _meta/*.json → write docs/*/benchmarks/*.md
#   Step 4  docs-en / docs-zh    Sphinx build → docs/*/build/html/
#
# WHAT IS AFFECTED:
#   docs-update        Writes evalscope/benchmarks/_meta/<name>.json (metadata only)
#   docs-update-stats  Same as above + downloads dataset to compute sample statistics
#   docs-translate     Updates readme.zh field inside each _meta/<name>.json
#   docs-generate      Overwrites docs/en/benchmarks/*.md + docs/zh/benchmarks/*.md
#
# PARAMETERS:
#   BENCHMARK  Specific benchmark name (e.g. gsm8k, mmlu).
#              Omit to process ALL registered benchmarks.
#   FORCE=1    Force re-translate even when a translation already exists.
#              Only applies to docs-translate.
#   WORKERS    Parallel worker count for update / translate (default: 4).
#
# COMMON USAGE:
#   make docs                               # Full pipeline: translate → generate → build HTML
#   make docs-update                        # Update metadata for ALL benchmarks
#   make docs-update BENCHMARK=gsm8k        # Update metadata for ONE benchmark
#   make docs-update-stats                  # Update metadata + stats for ALL benchmarks
#   make docs-update-stats BENCHMARK=gsm8k  # Update metadata + stats for ONE benchmark
#   make docs-translate                     # Translate only untranslated benchmarks (ALL)
#   make docs-translate BENCHMARK=gsm8k     # Translate ONE benchmark (skip if done)
#   make docs-translate FORCE=1             # Force re-translate ALL benchmarks
#   make docs-translate BENCHMARK=gsm8k FORCE=1  # Force re-translate ONE benchmark
#   make docs-generate                      # Regenerate .md files from persisted JSON data
#   make docs-en                            # Build English HTML docs only
#   make docs-zh                            # Build Chinese HTML docs only
#
# ============================================================================

# Parameters
BENCHMARK ?=
FORCE     ?=
WORKERS   ?= 4

# Internal helpers
# When BENCHMARK is set: pass it as positional arg; otherwise use --all flag
_BENCH_ARGS = $(if $(BENCHMARK),$(BENCHMARK),--all)
# When FORCE is non-empty (e.g. FORCE=1): append --force flag
_FORCE_FLAG = $(if $(FORCE),--force,)

.PHONY: docs
docs: docs-translate docs-generate
	$(MAKE) docs-en
	$(MAKE) docs-zh

.PHONY: docs-update
docs-update:
	python -m evalscope.cli.cli benchmark-info $(_BENCH_ARGS) --update --workers $(WORKERS)

.PHONY: docs-update-stats
docs-update-stats:
	python -m evalscope.cli.cli benchmark-info $(_BENCH_ARGS) --update --compute-stats --workers $(WORKERS)

.PHONY: docs-translate
docs-translate:
	python -m evalscope.cli.cli benchmark-info $(_BENCH_ARGS) --translate $(_FORCE_FLAG) --workers $(WORKERS)

.PHONY: docs-generate
docs-generate:
	python -m evalscope.cli.cli benchmark-info --generate-docs

.PHONY: docs-en
docs-en:
	cd docs/en && make clean && make html

.PHONY: docs-zh
docs-zh:
	cd docs/zh && make clean && make html

# ============================================================================
# Development
# ============================================================================

.PHONY: lint
lint:
	pre-commit run --all-files

.PHONY: dev
dev:
	pip install -e '.[dev,perf,docs]'
	pip install pre-commit

.PHONY: install
install:
	pip install -e .
