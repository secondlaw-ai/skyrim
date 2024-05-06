.PHONY: build install install-core lint test-publish publish clean format test test-unit test-integration

build:
	rm -rf ./dist/** *.egg-info && pip install -e . &&  python -m build
install:
	pip install -e '.[dev]'
install-core:
	pip install -e '.[dev]' && pip install -r requirements.txt
lint:
	ruff --fix .
test-publish:
	twine upload -r testpypi dist/*
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