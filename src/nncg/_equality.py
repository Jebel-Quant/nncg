"""Equality-augmented reduction: the Schur-complement saddle solve on a free set.

Factored out of :meth:`nncg.solver.ActiveSetSolver.solve_eq` so the outer loop
keeps only its orchestration. On a free set the saddle system for ``B x = c`` is
solved by eliminating the multiplier ``lambda`` in R^p through the p-by-p Schur
complement ``S = B_F A_F^{-1} B_F^T``: the ``p + 1`` right-hand sides share the
operator ``A_F`` and are each one inner solve, then ``S lambda = c - B_F v0``
fixes the multipliers in closed form.

Also home to the equality-specific input check (:func:`_require_eq_shapes`) and
the feasibility certificate (:func:`_eq_feasible`) that ``solve_eq`` adds on top
of the KKT exit: a rank-deficient, inconsistent ``B`` can pass the KKT test
while ``B x != c``, and must not be reported as converged.
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
    When ``B_F`` is rank-deficient ``S`` is singular and the multipliers fall back
    to the least-squares solution; the caller's feasibility check then decides
    whether ``B x = c`` actually holds.

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
    schur = b_f @ v1  # p-by-p Schur complement, SPD when B_F has full row rank
    rhs = c_eq - b_f @ v0
    try:
        lam = cholesky_solve(schur, rhs)
    except np.linalg.LinAlgError:
        # Rank-deficient B_F: take the least-squares multipliers and leave the
        # verdict on B x = c to _eq_feasible rather than crashing the solve.
        lam = np.asarray(np.linalg.lstsq(schur, rhs, rcond=None)[0], dtype=np.float64)
    xf = v0 + v1 @ lam  # x_F = A_F^{-1}(b_F + B_F^T lambda)
    return xf, lam, k0 + k_cols


def _require_eq_shapes(b_eq: Matrix, c_eq: Vector, n: int) -> None:
    """Validate that ``B`` has shape ``(p, n)`` and ``c`` has shape ``(p,)``.

    Args:
        b_eq: Equality matrix ``B``.
        c_eq: Equality right-hand side ``c``.
        n: Problem dimension ``len(b)``.

    Raises:
        ValueError: When ``b_eq`` is not two-dimensional with ``n`` columns, or
            ``c_eq`` is not one-dimensional with one entry per row of ``b_eq``.
    """
    b_shape, c_shape = np.shape(b_eq), np.shape(c_eq)
    if len(b_shape) != 2 or b_shape[1] != n:
        msg = f"b_eq must have shape (p, {n}), got {b_shape}"
        raise ValueError(msg)
    if c_shape != (b_shape[0],):
        msg = f"c_eq must have shape ({b_shape[0]},) to match b_eq {b_shape}, got {c_shape}"
        raise ValueError(msg)


def _eq_feasible(b_eq: Matrix, c_eq: Vector, x: Vector, tol: float) -> bool:
    """Return whether ``x`` satisfies ``B x = c`` to the relative tolerance ``tol``.

    With ``B_F`` of full row rank the Schur-complement solve makes ``B x = c``
    hold to rounding, so this only fails when the rank precondition is violated
    and ``c`` lies outside the range of ``B_F`` — the case the KKT test alone
    would certify.

    Args:
        b_eq: Equality matrix ``B`` of shape ``(p, n)``.
        c_eq: Equality right-hand side ``c`` of shape ``(p,)``.
        x: The candidate solution.
        tol: Tolerance on ``max|B x - c|``, relative to ``max(1, max|c|)``.

    Returns:
        True when the equality residual is within tolerance.
    """
    scale = max(1.0, float(np.max(np.abs(c_eq), initial=0.0)))
    return float(np.max(np.abs(b_eq @ x - c_eq), initial=0.0)) <= tol * scale
