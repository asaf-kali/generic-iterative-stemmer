LINE_LENGTH=120

install:
	pip install --upgrade pip
	pip install -r requirements.txt

commit:
	git add .
	git commit -m "Auto commit"
	git push

lint:
	black . -l $(LINE_LENGTH)
	isort . --profile black
	@make check-lint --no-print-directory

check-lint:
	black . -l $(LINE_LENGTH) --check
	isort . --profile black --check
	mypy . --ignore-missing-imports
	flake8 . --max-line-length=$(LINE_LENGTH)
