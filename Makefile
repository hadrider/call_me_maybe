PYTHON := python3.10
MYPY := mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

install:
	$(PYTHON) -m uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(PYTHON) -m flake8 src
	$(PYTHON) -m $(MYPY)

.PHONY: install run debug clean lint
