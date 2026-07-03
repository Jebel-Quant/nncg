"""Non-negative conjugate gradients: the active-set / block-principal-pivoting loop.

Solves the strictly convex non-negative quadratic program

    min_{x >= 0}  1/2 x^T A x - b^T x,        A symmetric positive definite,

and its equality-augmented variant with a general linear system ``B x = c``,
by wrapping a matrix-free conjugate-gradient inner solver in a primal-dual
active-set outer loop. The working-set toggles are the principal pivots of the
linear complementarity problem LCP(A, -b); guarding the fast block-pivot path
with a least-index Bland fallback gives unconditional finite termination at
the unique global minimiser — no non-degeneracy assumption (Theorem 5.1 of the
accompanying paper). See https://github.com/Jebel-Quant/mean_variance_solvers.

The quadratic term enters as a :class:`cvx.linalg.SymmetricOperator`, accessed
only through block products: ``apply_free`` drives the CG inner solves,
``matvec`` the reduced gradient, and ``solve_free`` the optional direct inner
solver. Wrap an explicit SPD array in ``DenseOperator``; for the Gram case
``A = M^T M + ridge I`` pass ``GramOperator(M, ridge)`` and the ``n x n``
matrix is never formed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cvx.linalg import Matrix, SymmetricOperator, Vector, cholesky_solve
from numpy.typing import NDArray

from .krylov import MatVec, cg, pcg

_NEEDS_OPERATOR = (
    "the quadratic term must be a cvx.linalg.SymmetricOperator: wrap a dense SPD "
    "array in DenseOperator(a), or pass GramOperator(M, ridge) for A = M'M + ridge*I"
)


def _require_operator(a: object) -> SymmetricOperator:
    """Validate that the quadratic term is a symmetric operator.

    Args:
        a: The candidate quadratic term.

    Returns:
        *a* unchanged when it is a :class:`cvx.linalg.SymmetricOperator`.

    Raises:
        TypeError: When *a* is anything else (e.g. a dense array).
    """
    if not isinstance(a, SymmetricOperator):
        raise TypeError(_NEEDS_OPERATOR)
    return a


def _free_matvec(op: SymmetricOperator, idx: NDArray[np.int_]) -> MatVec:
    """Return the free-block action ``v -> A[F, F] v`` of the operator.

    Args:
        op: The symmetric operator ``A``.
        idx: Integer positions of the free set ``F``.

    Returns:
        A callable computing ``op.apply_free(idx, v)``; the reduced matrix is
        never materialised.
    """
    return lambda v: op.apply_free(idx, v)


@dataclass(frozen=True)
class Result:
    """Outcome of an active-set solve.

    Attributes:
        x: The minimiser (or the final iterate if ``converged`` is False).
        outer: Number of outer active-set steps taken.
        inner: Total inner (CG/PCG) iterations across all outer steps; each
            direct inner solve counts as one.
        fallback: Number of least-index Bland fallback pivots taken.
        converged: True when the KKT exit was reached; False when an
            ``max_outer`` cap stopped the loop first.
        free: Boolean mask of the final free set.
        lam: Multipliers of the equality constraints (equality-augmented
            solves only; None otherwise).
        traj: The sequence of visited free sets as index tuples when
            trajectory tracking was requested; None otherwise.
    """

    x: Vector
    outer: int
    inner: int
    fallback: int
    converged: bool
    free: NDArray[np.bool_]
    lam: Vector | None = None
    traj: list[tuple[int, ...]] | None = None


def kkt_violation(a: SymmetricOperator, b: Vector, x: Vector) -> float:
    """Maximum violation of the KKT system of ``min_{x>=0} 1/2 x'Ax - b'x``.

    Args:
        a: The SPD operator ``A`` (a :class:`cvx.linalg.SymmetricOperator`).
        b: The linear term ``b``.
        x: Candidate solution.

    Returns:
        ``max`` of the negativity violations of ``x`` and of the reduced
        gradient ``s = A x - b``, and of the complementarity products
        ``|x_i s_i|``. Zero certifies the unique global minimiser.
    """
    op = _require_operator(a)
    s = op.matvec(x) - b
    return float(
        max(
            np.max(-x, initial=0.0),
            np.max(-s, initial=0.0),
            np.max(np.abs(x * s), initial=0.0),
        )
    )


def solve_nnqp(
    a: SymmetricOperator,
    b: Vector,
    tol: float = 1e-8,
    cg_tol: float = 1e-10,
    p_max: int = 3,
    inner: str = "cg",
    track: bool = False,
    cg_maxit: int = 100_000,
    max_outer: int | None = None,
    warm: tuple[NDArray[np.bool_], Vector] | None = None,
) -> Result:
    """Minimise ``1/2 x^T A x - b^T x`` over ``x >= 0`` by the active-set loop.

    Each free-block solve is matrix-free CG on ``v -> op.apply_free(F, v)``;
    the reduced matrix is never materialised and ``A`` is never refactorised.
    The batch block-pivot fast path is guarded by a least-index Bland
    fallback, so termination at the unique global minimiser is unconditional
    — no non-degeneracy assumption.

    Args:
        a: The SPD operator ``A`` (a :class:`cvx.linalg.SymmetricOperator`) —
            ``DenseOperator`` for an explicit array, ``GramOperator(M, ridge)``
            for ``A = M^T M + ridge I`` whose Gram matrix is never formed.
        b: The linear term ``b``.
        tol: Threshold of the primal and dual violator tests.
        cg_tol: Relative residual tolerance of the inner solves. Keep it a
            couple of orders below ``tol`` so the inexact loop makes the same
            sign decisions as the exact one (Lemma 5.1 of the paper).
        p_max: Patience budget — non-improving batch steps tolerated before a
            fallback pivot. Any value gives finite termination.
        inner: ``"cg"`` (matrix-free), ``"pcg"`` (Jacobi-preconditioned from
            ``op.diag``), or ``"exact"`` (direct solve of each free block via
            ``op.solve_free``). Match the inner solver to the backend: pick
            ``"exact"`` when ``solve_free`` is structured and cheap — e.g.
            ``FactorOperator``'s Woodbury solve at ``O(|F| r^2)`` — and CG
            when only products are cheap (large dense ``A``, Gram factors
            with many rows).
        track: Record the free-set trajectory in ``Result.traj``.
        cg_maxit: Iteration cap per inner solve.
        max_outer: Optional cap on outer steps; when hit, the current iterate
            is returned with ``converged=False``.
        warm: Optional ``(free_mask, x_prev)`` pair from a previous solve.
            Starts the loop from that free set and warm-starts every CG call
            from the newest iterate — across a support-stable parameter step
            the loop then terminates in a single outer step.

    Returns:
        A :class:`Result`; ``converged`` is True iff the KKT system was
        satisfied to ``tol``, which certifies the unique global minimiser.

    Raises:
        TypeError: When ``a`` is not a :class:`cvx.linalg.SymmetricOperator`.
        NotImplementedError: When ``inner="pcg"`` meets a backend that does
            not expose ``diag`` (propagated from ``cvx.linalg``).
    """
    op = _require_operator(a)
    n = len(b)
    dinv: Vector | None = None  # Jacobi preconditioner, read off op.diag on first use
    if warm is None:
        free = np.ones(n, dtype=bool)  # F = {1..n} initially
        x_guess: Vector | None = None
    else:
        free = warm[0].copy()
        x_guess = warm[1]
    x = np.zeros(n)
    n_bar = n + 1
    patience = p_max
    outer = inner_total = fallback = 0
    traj: list[tuple[int, ...]] | None = [] if track else None
    converged = True

    while True:
        if max_outer is not None and outer >= max_outer:
            converged = False
            break
        idx = np.flatnonzero(free)
        if traj is not None:
            traj.append(tuple(idx.tolist()))
        x0 = x_guess[idx] if x_guess is not None else None
        if inner == "exact":
            xf, k_step = op.solve_free(idx, b[idx]), 1
        elif inner == "pcg":
            if dinv is None:
                dinv = 1.0 / op.diag
            xf, k_step = pcg(_free_matvec(op, idx), b[idx], dinv[idx], tol=cg_tol, maxit=cg_maxit)
        else:
            xf, k_step = cg(_free_matvec(op, idx), b[idx], tol=cg_tol, maxit=cg_maxit, x0=x0)
        outer += 1
        inner_total += k_step

        x = np.zeros(n)
        x[idx] = xf
        if x_guess is not None:
            x_guess = x  # warm mode: newest iterate seeds the next reduced solve
        s = op.matvec(x) - b  # reduced gradient s_i = (Ax)_i - b_i

        prim = np.flatnonzero(free & (x < -tol))  # D: free but negative
        dual = np.flatnonzero((~free) & (s < -tol))  # V: bound but s < 0
        viol = np.concatenate([prim, dual])
        n_viol = viol.size
        if n_viol == 0:
            break  # KKT satisfied -> unique global minimiser

        if n_viol < n_bar or patience > 0:  # fast path: progress, or patience remains
            if n_viol < n_bar:
                n_bar = n_viol
                patience = p_max
            else:
                patience -= 1
            free[prim] = False  # batch exchange: drop all D, add all V
            free[dual] = True
        else:  # anti-cycling fallback: single Bland least-index pivot
            fallback += 1
            i_star = int(viol.min())
            free[i_star] = not free[i_star]

    return Result(
        x=x,
        outer=outer,
        inner=inner_total,
        fallback=fallback,
        converged=converged,
        free=free,
        traj=traj,
    )


def solve_nnqp_eq(
    a: SymmetricOperator,
    b: Vector,
    b_eq: Matrix,
    c_eq: Vector,
    tol: float = 1e-8,
    cg_tol: float = 1e-10,
    p_max: int = 3,
) -> Result:
    """Solve ``min 1/2 x^T A x - b^T x`` subject to ``x >= 0`` and ``B x = c``.

    On each free set the saddle system is solved by eliminating the multiplier
    ``lambda`` in R^p through the p-by-p Schur complement
    ``S = B_F A_F^{-1} B_F^T``: the ``p + 1`` right-hand sides share the
    operator ``A_F`` and are each one matrix-free CG solve, then
    ``S lambda = c - B_F v0`` fixes the multipliers in closed form. The single
    normalisation ``1^T x = beta`` is the ``p = 1`` case. ``B`` must have full
    row rank on the visited free sets (automatic for ``p = 1``).

    Args:
        a: The SPD operator ``A`` (a :class:`cvx.linalg.SymmetricOperator`).
        b: The linear term ``b``.
        b_eq: Equality matrix ``B`` of shape ``(p, n)``, full row rank.
        c_eq: Equality right-hand side ``c`` of shape ``(p,)``.
        tol: Threshold of the primal and dual violator tests.
        cg_tol: Relative residual tolerance of the inner CG solves.
        p_max: Patience budget of the batch fast path.

    Returns:
        A :class:`Result` with the multipliers in ``lam``. The reduced
        gradient underlying the dual test is ``s = A x - b - B^T lam``.

    Raises:
        TypeError: When ``a`` is not a :class:`cvx.linalg.SymmetricOperator`.
    """
    op = _require_operator(a)
    n = len(b)
    p = b_eq.shape[0]
    free = np.ones(n, dtype=bool)
    x = np.zeros(n)
    lam = np.zeros(p)
    n_bar = n + 1
    patience = p_max
    outer = inner_total = fallback = 0

    while True:
        idx = np.flatnonzero(free)
        matvec_f = _free_matvec(op, idx)
        b_f = b_eq[:, idx]
        v0, k0 = cg(matvec_f, b[idx], tol=cg_tol)
        v1 = np.zeros((idx.size, p))
        k_cols = 0
        for j in range(p):
            v1[:, j], kj = cg(matvec_f, b_f[j], tol=cg_tol)
            k_cols += kj
        outer += 1
        inner_total += k0 + k_cols

        schur = b_f @ v1  # p-by-p Schur complement, SPD
        lam = cholesky_solve(schur, c_eq - b_f @ v0)
        xf = v0 + v1 @ lam  # x_F = A_F^{-1}(b_F + B_F^T lambda)
        x = np.zeros(n)
        x[idx] = xf
        s = op.matvec(x) - b - b_eq.T @ lam  # reduced gradient

        prim = np.flatnonzero(free & (x < -tol))
        dual = np.flatnonzero((~free) & (s < -tol))
        viol = np.concatenate([prim, dual])
        n_viol = viol.size
        if n_viol == 0:
            break
        if n_viol < n_bar or patience > 0:
            if n_viol < n_bar:
                n_bar = n_viol
                patience = p_max
            else:
                patience -= 1
            free[prim] = False
            free[dual] = True
        else:
            fallback += 1
            i_star = int(viol.min())
            free[i_star] = not free[i_star]

    return Result(
        x=x,
        outer=outer,
        inner=inner_total,
        fallback=fallback,
        converged=True,
        free=free,
        lam=lam,
    )
