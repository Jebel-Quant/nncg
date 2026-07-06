"""Matrix-free (preconditioned) conjugate gradients — the Krylov core.

:func:`pcg` solves an SPD system accessed only through a mat-vec callable — the
matrix is never required explicitly — and takes its preconditioner as a callable
``r -> M^{-1} r`` (``config.precond=None`` recovers plain CG). It is
operator-agnostic: the preconditioner *builders* that turn a
:class:`cvx.linalg.SymmetricOperator` into such a callable live in
:mod:`nncg.inner`, alongside the inner solvers that use them. Convergence is
governed by the spectral condition number of ``M^{-1} A`` at the
``O(sqrt(kappa))`` Krylov rate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from cvx.linalg import Vector

MatVec = Callable[[Vector], Vector]

#: A preconditioner is the action ``r -> M^{-1} r`` of an SPD preconditioner
#: ``M ~ A`` — a general linear map, not necessarily diagonal. PCG only ever
#: multiplies by ``M^{-1}``, so the callable is all it needs. Build one from an
#: operator with the inner solvers in :mod:`nncg.inner`.
Preconditioner = Callable[[Vector], Vector]


@dataclass(frozen=True)
class KrylovConfig:
    """Options for a preconditioned CG solve (:func:`pcg`).

    Bundles the solve knobs into one argument so :func:`pcg` keeps a short
    signature (matrix and right-hand side, then the config).

    Attributes:
        precond: The action ``r -> M^{-1} r`` of an SPD preconditioner, applied
            once per iteration; ``None`` is the identity, so PCG reduces to
            plain CG.
        tol: Relative residual stopping tolerance ``||b - A x|| / ||b||``.
        maxit: Iteration cap; the current iterate is returned when it is hit.
        x0: Optional warm start. The initial residual is ``b - A x0``, so a good
            guess cuts the iteration count by the log of the initial error.
    """

    precond: Preconditioner | None = None
    tol: float = 1e-8
    maxit: int = 100_000
    x0: Vector | None = None


#: Shared default so ``KrylovConfig()`` is not called in argument defaults (ruff B008).
_DEFAULT_KRYLOV = KrylovConfig()

# Why in-house rather than scipy.sparse.linalg.cg? scipy's CG is matrix-free
# and preconditionable too, so functionally it could stand in here. We keep our
# own for four reasons: (1) the matrix-free PCG is this package's core
# contribution — the reference implementation of the paper — and must stay
# auditable, not delegated to a black box; (2) the runtime dependency set is
# NumPy + cvx-linalg only, and this is ~60 lines NumPy already covers; (3) we
# return the iteration count, which the numerical study asserts on, whereas
# scipy returns only a convergence flag (recovering the count needs a callback);
# (4) scipy's atol/rtol stopping semantics have shifted across releases, so
# owning the loop pins the exact criterion and keeps the paper's numbers stable.
# Third-party solvers belong in the baseline comparisons (kept in the paper
# repo, Jebel-Quant/mean_variance_solvers), not in this inner Krylov core.


def pcg(
    matvec: MatVec,
    rhs: Vector,
    config: KrylovConfig = _DEFAULT_KRYLOV,
) -> tuple[Vector, int]:
    """Solve an SPD system by preconditioned conjugate gradients.

    PCG converges at the condition number of ``M^{-1} A`` rather than of ``A``,
    where the preconditioner ``M^{-1}`` (``config.precond``) enters only as the
    action ``r -> M^{-1} r``; ``config.precond=None`` is the identity, so PCG
    reduces to plain CG. The inner solvers in :mod:`nncg.inner` build suitable
    preconditioners from an operator (diagonal Jacobi, randomized Nyström).

    Args:
        matvec: The action ``v -> A v`` of an SPD operator.
        rhs: Right-hand side ``b``.
        config: Preconditioner, tolerance, iteration cap and warm start of the
            solve (see :class:`KrylovConfig`).

    Returns:
        The approximate solution and the number of iterations taken.
    """
    precond, tol, maxit, x0 = config.precond, config.tol, config.maxit, config.x0
    if precond is None:
        precond = lambda r: r  # noqa: E731 -- identity preconditioner; PCG becomes plain CG
    if x0 is None:
        x = np.zeros_like(rhs)
        r = rhs.copy()
    else:
        x = x0.astype(np.float64, copy=True)
        r = rhs - matvec(x)
    z = precond(r)
    p = z.copy()
    rz = float(r @ z)
    bnorm = float(np.linalg.norm(rhs))
    if bnorm == 0.0:
        return np.zeros_like(rhs), 0
    # A warm start that already solves the system to tolerance leaves r == 0,
    # so p == 0 and the search-direction curvature p @ ap vanishes; returning
    # here avoids the 0/0 in the alpha step and reports zero iterations.
    if float(np.linalg.norm(r)) / bnorm <= tol:
        return x, 0
    for it in range(1, maxit + 1):
        ap = matvec(p)
        alpha = rz / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        if float(np.linalg.norm(r)) / bnorm <= tol:
            return x, it
        z = precond(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxit
