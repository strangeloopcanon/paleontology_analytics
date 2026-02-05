.PHONY: setup bootstrap check test all deps-audit llm-live release

setup bootstrap:
	@echo "Using uv-managed runtime; install requirements with: uv pip install -r requirements.txt"

check:
	uvx ruff check src thesis tests --output-format=full

test:
	uv run pytest -q

deps-audit:
	@echo "Running advisory dependency audit (non-blocking in baseline mode)"
	@if command -v pip-audit >/dev/null 2>&1; then \
		pip-audit -r requirements.txt; \
	else \
		uvx pip-audit -r requirements.txt || true; \
	fi

llm-live:
	@echo "No LLM live test target is configured for this repository."

all: check test

release:
	@echo "Release automation is not configured in this repository."
