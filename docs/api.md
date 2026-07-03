# API reference

The public API: the two solvers and the KKT certificate. The matrix-free
inner solvers in `nncg.krylov` are documented below as the package's core
contribution, but are driven by the solvers rather than imported directly.
The planted-optimum problem generators live outside the installed package,
in the repository's `tests/problems.py`.

## Solver

::: nncg.solver

## Inner solvers (internal)

::: nncg.krylov
