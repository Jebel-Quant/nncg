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

- `src/nncg/solver.py` — the active-set loop: `ActiveSetSolver` (the outer-loop
  orchestrator) and its `ActiveSetConfig`, the `InnerSolver` protocol, and the
  `Result` dataclass.
- `src/nncg/_active_set.py` — the pure active-set primitives the loop is built
  from: free-set seeding, the primal/dual violator split, the guarded batch/Bland
  `_pivot`, and the `_drive` driver loop.
- `src/nncg/_equality.py` — the equality-augmented `_saddle_solve` (the p-by-p
  Schur-complement elimination behind `ActiveSetSolver.solve_eq`).
- `src/nncg/certificate.py` — the public `kkt_violation` certificate and the
  shared `_require_operator` precondition.
- `src/nncg/inner.py` — the built-in inner solvers (`CG`, `Jacobi`, `Nystrom`,
  `Exact`).
- `src/nncg/preconditioners.py` — the free-block operator and preconditioner
  builders the inner solvers run on (`_free_matvec`, `_jacobi`, and the
  randomized-Nyström machinery plus `NystromConfig`).
- `src/nncg/api.py` — the one-call convenience wrappers `solve_nnqp` /
  `solve_nnqp_eq` over `ActiveSetSolver`; logic-free, they just wrap arrays in
  `DenseOperator` and resolve the `inner` string shortcut.
- `src/nncg/krylov.py` — plain and Jacobi-preconditioned CG, warm-startable.
- `tests/problems.py` — planted-optimum generators, including the adversarial
  anti-correlated family that forces the Bland fallback. Deliberately outside
  the installed package; also intended for later experiments and notebooks.
- `tests/` — the paper's numerical study as a test suite. Keep it that way:
  every mathematical claim in the paper that this package implements should
  have a test here, and the Bland/fallback path must stay exercised (currently
  in `tests/test_nncg/test_solver.py`, driven by the adversarial anti-correlated
  family in `tests/problems.py`) — it is the termination guarantee's load-bearing
  component.

## Test layout — an accepted deviation from Rhiza 1:1 parity

The tests use an **intentional grouped layout**, not the Rhiza convention of one
mirrored `tests/nncg/test_<module>.py` per `src/nncg/<module>.py` that the
template's `scripts/check_test_layout.py` enforces:

- `tests/test_nncg/` — behaviour tests grouped by concern (`test_api`,
  `test_inner`, `test_krylov`, `test_solver`).
- `tests/test_paper/` — the paper's numerical study (`test_cg_convergence`,
  `test_conditioning`, `test_reduction`, `test_results`).
- `tests/problems.py` — shared planted-optimum generators (see above).

Several internal primitives (`_active_set.py`, `_equality.py`, `certificate.py`)
have no dedicated 1:1 test file **by design**: they are the pieces the active-set
loop is built from, so they are exercised end-to-end through
`tests/test_nncg/test_solver.py` rather than in isolation. This keeps the suite
organised around the paper's claims and the solver's observable behaviour.

This is a **deliberate, accepted deviation** — treat it as by-design, not a gap.
The layout gives full line coverage of every source module, and
`check_test_layout.py` is not wired into any gate in this repo. If you add a new
public entry point, add its tests to the matching grouped file (or a new one
under `tests/test_nncg/` or `tests/test_paper/`); do not reshuffle the suite into
per-module files to satisfy the parity check.

## Conventions

- Runtime dependencies are NumPy and `cvx-linalg` (the Jebel-Quant linear
  algebra package). Do not add others.
- The solvers take the quadratic term exclusively as a
  `cvx.linalg.SymmetricOperator` (`DenseOperator` for explicit arrays,
  `GramOperator` for `A = M'M + ridge I`): `restricted` provides the
  pre-sliced free block that drives the in-house CG, `matvec` the reduced
  gradient, `solve_free` only the `inner="exact"` path. The matrix-free CG/PCG in `krylov.py` stays in-house — it is this
  package's core contribution; do not swap it for `bordered_solve` or other
  direct factorisations.
- All public functions carry full docstrings and type hints (CI gates on
  both).
- `make test`, `make fmt`, `make typecheck`, `make deptry` — see `make help`
  for the full menu.
