# default rule
default: install

.PHONY: docs
docs: docs-en docs-zh

.PHONY: docs-en
docs-en:
	cd docs/en && make clean && make html

.PHONY: docs-zh
docs-zh:
	cd docs/zh && make clean && make html

.PHONY: linter
linter:
	pre-commit run --all-files

.PHONY: install
install:
	pip install -e .'[dev,perf,docs]'
