# Paleontology Analytics

So what: this repo contains a small interactive dashboard plus a set of reproducible research analyses exploring macro-scale paleobiological patterns (e.g., functional convergence vs climate volatility).

## Start here

- Dashboard: `dashboard/` (static site; JSON artifacts are tracked)
- Research writeups + figures: `thesis/README.md`

## Development gates

- `make check`: Ruff linting for `src/`, `thesis/`, and `tests/`
- `make test`: pytest regression suite (`tests/`)
- `make all`: `check` then `test`
- `make deps-audit`: advisory `pip-audit` run (baseline mode)

## License

Apache-2.0 (see `LICENSE`).

## Citation

If you use this repository, please cite it (see `CITATION.cff`).
