# CLAUDE.md

Guidance for Claude Code (and human contributors) working in this repository.

## What this project is

`nncg` implements **non-negative conjugate gradients**: the bound-constrained
SPD quadratic `min_{x>=0} 1/2 x'Ax - b'x` (and its equality-augmented variant
`Bx = c`) solved by a matrix-free CG inner solver inside a primal-dual
active-set loop with an unconditional finite-termination guarantee. It is the
reference implementation of the paper "Non-Negative Conjugate Gradients"
(Schmelzer & Stoll), developed in
[Jebel-Quant/mean_variance_solvers](https://github.com/Jebel-Quant/mean_variance_solvers).
The library lives in `src/nncg/`; everything else is tests, docs, or
development infrastructure synced from the Rhiza template.

## The Rhiza split — read this before editing config

This repo syncs its development infrastructure (CI workflows, Makefile,
linters, test harness, release tooling) from the **mother repo
`jebel-quant/rhiza`**. The pin lives in `.rhiza/template.yml`. Do not hand-edit
synced files (`.rhiza/**`, `.github/workflows/rhiza_*.yml`, `ruff.toml`, ...):
changes are overwritten on the next sync. Repo-owned files are `Makefile`,
`pyproject.toml`, `README.md`, `CLAUDE.md`, `src/**`, `tests/**`, `docs/**`.

## Layout

- `src/nncg/solver.py` — the active-set loop (`solve_nnqp`, `solve_nnqp_eq`),
  the `Result` dataclass, and the `kkt_violation` certificate.
- `src/nncg/krylov.py` — plain and Jacobi-preconditioned CG, warm-startable.
- `src/nncg/problems.py` — planted-optimum generators, including the
  adversarial anti-correlated family that forces the Bland fallback.
- `tests/` — the paper's numerical study as a test suite. Keep it that way:
  every mathematical claim in the paper that this package implements should
  have a test here, and `tests/test_fallback.py` must keep the fallback path
  exercised (it is the termination guarantee's load-bearing component).

## Conventions

- Runtime dependencies are NumPy and `cvx-linalg` (the Jebel-Quant linear
  algebra package: shared `Matrix`/`Vector` aliases, `cholesky_solve` for the
  SPD direct solves). Do not add others. The matrix-free CG/PCG in
  `krylov.py` stays in-house: `cvx-linalg`'s `solve_free`/`bordered_solve`
  are direct factorisations, and the CG inner solver is this package's core
  contribution.
- All public functions carry full docstrings and type hints (CI gates on
  both).
- `make test`, `make fmt`, `make typecheck`, `make deptry` — see `make help`
  for the full menu.
