# default rule
default: install

.PHONY: docs
docs:
	python -m docs.scripts.translate_description
	python -m docs.scripts.generate_dataset_md
	$(MAKE) docs-en
	$(MAKE) docs-zh

.PHONY: docs-en
docs-en:
	cd docs/en && make clean && make html

.PHONY: docs-zh
docs-zh:
	cd docs/zh && make clean && make html

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
