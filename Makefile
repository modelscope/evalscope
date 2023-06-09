WHL_BUILD_DIR :=package
DOC_BUILD_DIR :=docs/build/

# default rule
default: whl docs

.PHONY: docs

.PHONY: linter
linter:
	bash .dev_scripts/linter.sh

.PHONY: whl
whl:
	python setup.py sdist bdist_wheel

.PHONY: clean
clean:
	rm -rf  $(WHL_BUILD_DIR) $(DOC_BUILD_DIR)
