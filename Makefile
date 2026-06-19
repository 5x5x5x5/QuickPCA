.PHONY: install test lint format typecheck docs build clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check quickpca tests

format:
	ruff format quickpca tests

typecheck:
	mypy quickpca

docs:
	mkdocs build

build:
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov site
	find . -type d -name __pycache__ -exec rm -rf {} +
