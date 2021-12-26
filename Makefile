LINE_LENGTH=120

install:
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt

commit:
	git add .
	git commit -m "Auto commit"
	git push

lint:
	black . -l $(LINE_LENGTH)
	isort . --profile black --skip __init__.py
	@make check-lint --no-print-directory

check-lint:
	black . -l $(LINE_LENGTH) --check
	isort . --profile black --check --skip __init__.py
	mypy . --ignore-missing-imports
	flake8 . --max-line-length=$(LINE_LENGTH)
