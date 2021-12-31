.PHONY: tests

LINE_LENGTH=120

install:
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	black . -l $(LINE_LENGTH)
	isort . --profile black --skip __init__.py
	@make check-lint --no-print-directory

check-lint:
	black . -l $(LINE_LENGTH) --check
	isort . --profile black --check --skip __init__.py
	mypy . --ignore-missing-imports
	flake8 . --max-line-length=$(LINE_LENGTH)

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
