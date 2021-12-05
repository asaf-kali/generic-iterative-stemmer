install:
	pip install --upgrade pip
	pip install -r requirements.txt

commit:
	git add .
	git commit -m "Auto commit"
	git push
