# API reference

Every name exported from the `nncg` namespace, grouped the way you reach for
them: the one-call wrappers first, then the two solver families they wrap
(the active-set loop and MPRGP), then the KKT certificate that both are
judged by, and finally the inner solvers and the matrix-free Krylov core the
active-set loop runs on.

Private helpers are omitted. The planted-optimum problem generators live
outside the installed package, in the repository's `tests/problems.py`.

## One-call wrappers

Logic-free shortcuts over the solver classes below: they wrap a plain array in
a `DenseOperator` and resolve the `inner` string. Reach for these first.

::: nncg.api

## Active-set solver

The primal-dual active-set loop with the unconditional finite-termination
guarantee — this package's subject. `solve_eq` adds the equality-augmented
`Bx = c` variant via a p-by-p Schur complement.

::: nncg.solver

## MPRGP

A standalone matrix-free projection solver for the same bound-constrained
problem (Dostál & Schöberl) — conjugate-gradient, expansion and proportioning
steps under the proportioning test, no factorisation. A first-order
alternative to the active-set loop; bound constraints only.

::: nncg.mprgp

## KKT certificate

::: nncg.certificate

## Inner solvers

The pluggable free-block solvers the active-set loop delegates to. Pass an
instance to `ActiveSetSolver(inner=...)` to tune one; the string shortcuts on
the wrappers take defaults only.

::: nncg.inner

## Krylov core

The in-house matrix-free CG and Jacobi-preconditioned CG, warm-startable.
This is the package's core contribution — the inner solvers drive it rather
than you calling it directly.

::: nncg.krylov
