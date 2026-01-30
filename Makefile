# default rule
default: install

# ============================================================================
# Documentation Generation
# ============================================================================
# Usage:
#   make docs                      # Generate docs (translate if needed, then build)
#   make docs-update               # Update all benchmark data (metadata only)
#   make docs-update-stats         # Update all benchmark data with statistics
#   make docs-update BENCHMARK=gsm8k  # Update specific benchmark
#   make docs-translate            # Translate benchmarks that need it
#   make docs-translate FORCE=1    # Force re-translate all
#   make docs-generate             # Generate docs from persisted data
# ============================================================================

# Parameters
BENCHMARK ?=
FORCE ?=
WORKERS ?= 4

.PHONY: docs
docs: docs-translate docs-generate
	$(MAKE) docs-en
	$(MAKE) docs-zh

.PHONY: docs-update
docs-update:
ifdef BENCHMARK
	python -m evalscope.cli.cli benchmark-info $(BENCHMARK) --update
else
	python -m evalscope.cli.cli benchmark-info --all --update
endif

.PHONY: docs-update-stats
docs-update-stats:
ifdef BENCHMARK
	python -m evalscope.cli.cli benchmark-info $(BENCHMARK) --update --compute-stats
else
	python -m evalscope.cli.cli benchmark-info --all --update --compute-stats
endif

.PHONY: docs-translate
docs-translate:
ifdef BENCHMARK
ifdef FORCE
	python -m evalscope.cli.cli benchmark-info $(BENCHMARK) --translate --force --workers $(WORKERS)
else
	python -m evalscope.cli.cli benchmark-info $(BENCHMARK) --translate --workers $(WORKERS)
endif
else
ifdef FORCE
	python -m evalscope.cli.cli benchmark-info --translate --force --workers $(WORKERS)
else
	python -m evalscope.cli.cli benchmark-info --translate --workers $(WORKERS)
endif
endif

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
