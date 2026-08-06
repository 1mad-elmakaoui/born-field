.PHONY: install lint fmt type test check clean

install:
	uv sync --all-extras --dev

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

type:
	uv run mypy

test:
	uv run pytest --cov=born_field --cov-report=term-missing

# What CI runs. Run this before pushing.
check: lint type test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage
