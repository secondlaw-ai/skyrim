build:
	rm -rf ./dist/** && python -m build
install:
	pip install -e '.[dev]'
install-core:
	pip install -e '.[dev]' && pip install -r requirements.txt
test-publish:
	twine upload -r testpypi dist/*
lint:
	ruff --fix .
publish:
	twine upload -r pypi dist/*
clean:
	rm -rf dist/ build/ *.egg-info
format:
	black .
test:
	pytest tests
test-unit:
	pytest --ignore=tests/integration
test-integration:
	pytest tests/integration