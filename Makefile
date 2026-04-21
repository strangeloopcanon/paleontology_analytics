.PHONY: setup bootstrap check test all deps-audit llm-live release

# Use uv wrappers locally; fall back to bare commands in CI.
RUN := $(if $(shell command -v uv 2>/dev/null),uv run,)
RUFF := $(if $(shell command -v uvx 2>/dev/null),uvx ruff,ruff)

setup bootstrap:
	@echo "Using uv-managed runtime; install requirements with: uv pip install -r requirements.txt"

check:
	$(RUFF) check src thesis tests --output-format=full
	$(RUN) mypy --ignore-missing-imports --disable-error-code=import-untyped src
	bandit -r src -ll || true

test:
	$(RUN) pytest -q --cov=src --cov-report=term-missing

deps-audit:
	@echo "Running advisory dependency audit (non-blocking in baseline mode)"
	@if command -v pip-audit >/dev/null 2>&1; then \
		pip-audit -r requirements.txt; \
	elif command -v uvx >/dev/null 2>&1; then \
		uvx pip-audit -r requirements.txt || true; \
	else \
		echo "pip-audit not found; skipping"; \
	fi

llm-live:
	@echo "No LLM live test target is configured for this repository."

all: check test

release:
	@echo "Release automation is not configured in this repository."
