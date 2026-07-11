"""Equality-augmented reduction: the Schur-complement saddle solve on a free set.

Factored out of :meth:`nncg.solver.ActiveSetSolver.solve_eq` so the outer loop
keeps only its orchestration. On a free set the saddle system for ``B x = c`` is
solved by eliminating the multiplier ``lambda`` in R^p through the p-by-p Schur
complement ``S = B_F A_F^{-1} B_F^T``: the ``p + 1`` right-hand sides share the
operator ``A_F`` and are each one inner solve, then ``S lambda = c - B_F v0``
fixes the multipliers in closed form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cvx.linalg import Matrix, SymmetricOperator, Vector, cholesky_solve
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .solver import InnerSolver


def _saddle_solve(
    inner: InnerSolver,
    a: SymmetricOperator,
    b: Vector,
    b_eq: Matrix,
    c_eq: Vector,
    idx: NDArray[np.int_],
    x0: Vector | None,
) -> tuple[Vector, Vector, int]:
    """Solve the equality-augmented saddle system on the free set ``idx``.

    Runs ``p + 1`` inner solves through the shared free-block operator ``A_F``
    (the ``v0`` column warm-started at ``x0``, the ``v1`` columns cold), forms the
    SPD Schur complement ``S = B_F A_F^{-1} B_F^T`` and recovers the multipliers
    from ``S lambda = c - B_F v0`` before back-substituting ``x_F = v0 + v1 lambda``.

    Args:
        inner: The inner solver driving each free-block solve.
        a: The SPD operator ``A``.
        b: The linear term ``b``.
        b_eq: Equality matrix ``B`` of shape ``(p, n)``, full row rank on ``idx``.
        c_eq: Equality right-hand side ``c`` of shape ``(p,)``.
        idx: Integer positions of the free set ``F``.
        x0: Warm inner guess for the ``v0`` column restricted to ``idx``, or ``None``.

    Returns:
        ``(x_F, lam, inner_iters)``: the free-block solution, the equality
        multipliers, and the total inner iteration count across all columns.
    """
    p = b_eq.shape[0]
    b_f = b_eq[:, idx]
    v0, k0 = inner.solve(a, idx, b[idx], x0)
    v1 = np.zeros((idx.size, p))
    k_cols = 0
    for j in range(p):
        v1[:, j], kj = inner.solve(a, idx, b_f[j], None)
        k_cols += kj
    schur = b_f @ v1  # p-by-p Schur complement, SPD
    lam = cholesky_solve(schur, c_eq - b_f @ v0)
    xf = v0 + v1 @ lam  # x_F = A_F^{-1}(b_F + B_F^T lambda)
    return xf, lam, k0 + k_cols
