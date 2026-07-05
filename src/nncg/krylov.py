"""Conjugate-gradient inner solvers.

Plain CG and Jacobi-preconditioned CG for symmetric positive definite (SPD)
systems, accessed only through a mat-vec callable — the matrix is never
required explicitly. These are the inner solvers of the active-set loop in
:mod:`nncg.solver`; their convergence is governed by the spectral condition
number kappa at the O(sqrt(kappa)) Krylov rate.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from cvx.linalg import Vector

MatVec = Callable[[Vector], Vector]

# Why in-house rather than scipy.sparse.linalg.cg? scipy's CG is matrix-free
# and preconditionable too, so functionally it could stand in here. We keep our
# own for four reasons: (1) the matrix-free CG/PCG is this package's core
# contribution — the reference implementation of the paper — and must stay
# auditable, not delegated to a black box; (2) the runtime dependency set is
# NumPy + cvx-linalg only, and this is ~90 lines NumPy already covers; (3) we
# return the iteration count, which the numerical study asserts on, whereas
# scipy returns only a convergence flag (recovering the count needs a callback);
# (4) scipy's atol/rtol stopping semantics have shifted across releases, so
# owning the loop pins the exact criterion and keeps the paper's numbers stable.
# Third-party solvers belong in the baseline comparisons (kept in the paper
# repo, Jebel-Quant/mean_variance_solvers), not in this inner Krylov core.


def cg(
    matvec: MatVec,
    rhs: Vector,
    tol: float = 1e-8,
    maxit: int = 100_000,
    x0: Vector | None = None,
) -> tuple[Vector, int]:
    """Solve an SPD system by conjugate gradients.

    Args:
        matvec: The action ``v -> A v`` of an SPD operator.
        rhs: Right-hand side ``b``.
        tol: Relative residual stopping tolerance ``||b - A x|| / ||b||``.
        maxit: Iteration cap; the current iterate is returned when it is hit.
        x0: Optional warm start. The initial residual is ``b - A x0``, so a
            good guess cuts the iteration count by the log of the initial
            error.

    Returns:
        The approximate solution and the number of iterations taken.
    """
    if x0 is None:
        x = np.zeros_like(rhs)
        r = rhs.copy()
    else:
        x = x0.astype(np.float64, copy=True)
        r = rhs - matvec(x)
    p = r.copy()
    rs = float(r @ r)
    bnorm = float(np.linalg.norm(rhs))
    if bnorm == 0.0:
        return np.zeros_like(rhs), 0
    # A warm start that already solves the system to tolerance leaves r == 0,
    # so p == 0 and the search-direction curvature p @ ap vanishes; returning
    # here avoids the 0/0 in the alpha step and reports zero iterations.
    if np.sqrt(rs) / bnorm <= tol:
        return x, 0
    for it in range(1, maxit + 1):
        ap = matvec(p)
        alpha = rs / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) / bnorm <= tol:
            return x, it
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, maxit


def pcg(
    matvec: MatVec,
    rhs: Vector,
    dinv: Vector,
    tol: float = 1e-8,
    maxit: int = 100_000,
    x0: Vector | None = None,
) -> tuple[Vector, int]:
    """Solve an SPD system by Jacobi-preconditioned conjugate gradients.

    The preconditioner is ``M^{-1} = diag(dinv)``; for operators that are a
    well-conditioned core under a bad diagonal scaling, PCG runs at the core's
    condition number regardless of the scaling.

    Args:
        matvec: The action ``v -> A v`` of an SPD operator.
        rhs: Right-hand side ``b``.
        dinv: Elementwise inverse of the operator's diagonal.
        tol: Relative residual stopping tolerance.
        maxit: Iteration cap; the current iterate is returned when it is hit.
        x0: Optional warm start. The initial residual is ``b - A x0``, so a
            good guess cuts the iteration count by the log of the initial
            error.

    Returns:
        The approximate solution and the number of iterations taken.
    """
    if x0 is None:
        x = np.zeros_like(rhs)
        r = rhs.copy()
    else:
        x = x0.astype(np.float64, copy=True)
        r = rhs - matvec(x)
    z = dinv * r
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
        z = dinv * r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxit
