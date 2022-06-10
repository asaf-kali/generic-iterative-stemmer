.PHONY: tests

LINE_LENGTH=120

# Install

install-run:
	pip install --upgrade pip
	pip install -r requirements.txt

install-test:
	@make install-run --no-print-directory
	pip install -r requirements-dev.txt

install-dev:
	@make install-test --no-print-directory
	pre-commit install
	@make test --no-print-directory


install: install-dev

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
	python -m pytest -s

# Pypi

build:
	gio trash -f dist/
	gio trash -f generic_iterative_stemmer.egg-info/
	python setup.py sdist

upload:
	make build
	twine upload dist/*

upload-test:
	make build
	twine upload --repository testpypi dist/*
