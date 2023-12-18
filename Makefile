PYTHON_TEST_COMMAND=pytest -s
ifeq ($(OS),Windows_NT)
	OPEN_FILE_COMMAND=start
	DEL_COMMAND=del
else
	OPEN_FILE_COMMAND=xdg-open
	DEL_COMMAND=gio trash
endif
SYNC=--sync
.PHONY: tests build

LINE_LENGTH=120

# Install

upgrade-pip:
	python -m pip install --upgrade pip

install-run: upgrade-pip
	pip install -r requirements.txt

install-test: upgrade-pip
	pip install -r requirements-dev.txt
	@make install-run --no-print-directory

install-dev:
	@make install-test --no-print-directory
	pre-commit install

install: install-dev test

# Lint

lint-only:
	black . -l $(LINE_LENGTH)
	isort . --profile black --skip __init__.py

lint-check:
	black . -l $(LINE_LENGTH) --check
	isort . --profile black --skip __init__.py --check
	mypy . --ignore-missing-imports
	flake8 . --max-line-length=$(LINE_LENGTH)

lint: lint-only
	pre-commit run --all-files

# Test

test:
	python -m $(PYTHON_TEST_COMMAND)

# Pypi

build:
	$(DEL_COMMAND) -f dist/*
	python -m build

upload-only:
	twine upload dist/*

upload: build upload-only

upload-test:
	make build
	twine upload --repository testpypi dist/*
