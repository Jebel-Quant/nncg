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

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from cvx.linalg import Matrix, SymmetricOperator, Vector, cholesky_solve
from numpy.typing import NDArray

from .krylov import MatVec, cg, pcg

SubSolve = Callable[[NDArray[np.int_], "Vector | None"], "tuple[Vector, Vector | None, int]"]
"""Subproblem solve on a free set: ``(idx, x0) -> (x_F, lam, inner_iters)``."""

ReducedGradient = Callable[["Vector", "Vector | None"], "Vector"]
"""Reduced gradient of the subproblem: ``(x, lam) -> s``."""

_NEEDS_OPERATOR = (
    "the quadratic term must be a cvx.linalg.SymmetricOperator: wrap a dense SPD "
    "array in DenseOperator(a), or pass GramOperator(M, ridge) for A = M'M + ridge*I"
)
_RCOND_MIN = 1e-12  # matches cvx-linalg's DEFAULT_COND_THRESHOLD of 1e12


def _check_dimension(op: SymmetricOperator, b: Vector) -> None:
    """Check that the operator and the linear term agree in dimension.

    Args:
        op: The symmetric operator ``A``.
        b: The linear term ``b``.

    Raises:
        ValueError: When ``op.n != len(b)``.
    """
    if op.n != len(b):
        msg = f"operator dimension {op.n} does not match len(b) = {len(b)}"
        raise ValueError(msg)


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

    The free-set restriction is hoisted out of the inner loop: when the
    operator provides ``restricted`` (cvx-linalg >= 0.10), the pre-sliced
    free-block operator is built once here and the returned callable is its
    plain ``matvec``. Calling ``apply_free(idx, v)`` per CG iteration instead
    re-gathers the operator's storage (e.g. the Gram factor columns) on every
    call, which costs an order of magnitude more wall clock at identical
    iteration counts. The fallback keeps older cvx-linalg releases working.

    Args:
        op: The symmetric operator ``A``.
        idx: Integer positions of the free set ``F``.

    Returns:
        A callable computing ``A[F, F] @ v``; the reduced matrix is never
        materialised.
    """
    restricted = getattr(op, "restricted", None)
    if restricted is not None:
        try:
            return cast(MatVec, restricted(idx).matvec)
        except NotImplementedError:
            pass  # backend without a pre-sliced form; fall back below
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


def _active_set_loop(
    n: int,
    sub_solve: SubSolve,
    reduced_gradient: ReducedGradient,
    tol: float,
    p_max: int,
    track: bool = False,
    max_outer: int | None = None,
    warm: tuple[NDArray[np.bool_], Vector] | None = None,
) -> Result:
    """Run the guarded primal-dual active-set loop shared by both solvers.

    The driver owns everything the termination proof depends on: the primal
    and dual violator tests, the batch exchange with its patience counter,
    and the least-index Bland fallback. What is solved on each free set — a
    single reduced system, or the equality-augmented saddle system — enters
    through the ``sub_solve`` callback, with ``reduced_gradient`` supplying
    the matching dual test quantity.

    Args:
        n: Problem dimension.
        sub_solve: Callback ``(idx, x0) -> (x_F, lam, inner_iters)`` solving
            the subproblem on the free set ``idx``. ``x0`` is a warm inner
            guess restricted to ``idx`` (None on a cold start); ``lam`` are
            the equality multipliers (None for the bound-only problem).
        reduced_gradient: Callback ``(x, lam) -> s`` computing the reduced
            gradient that drives the dual violator test.
        tol: Threshold of the primal and dual violator tests.
        p_max: Patience budget — non-improving batch steps tolerated before a
            fallback pivot. Any value gives finite termination.
        track: Record the free-set trajectory in ``Result.traj``.
        max_outer: Optional cap on outer steps; when hit, the current iterate
            is returned with ``converged=False``.
        warm: Optional ``(free_mask, x_prev)`` pair from a previous solve.
            Starts the loop from that free set and seeds every subproblem
            solve from the newest iterate.

    Returns:
        A :class:`Result`; ``lam`` is whatever the last subproblem returned.
    """
    if warm is None:
        free = np.ones(n, dtype=bool)  # F = {1..n} initially
        x_guess: Vector | None = None
    else:
        free = warm[0].copy()
        x_guess = warm[1]
    x = np.zeros(n)
    lam: Vector | None = None
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
        xf, lam, k_step = sub_solve(idx, x0)
        outer += 1
        inner_total += k_step

        x = np.zeros(n)
        x[idx] = xf
        if x_guess is not None:
            x_guess = x  # warm mode: newest iterate seeds the next reduced solve
        s = reduced_gradient(x, lam)

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
        lam=lam,
        traj=traj,
    )


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
    _check_dimension(op, b)
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
        ValueError: When the operator dimension does not match ``len(b)``, or
            when ``inner="exact"`` meets a numerically singular free block
            (``op.rcond_free`` below 1e-12) — ``A`` is then not positive
            definite on that free set; add a ridge.
        NotImplementedError: When ``inner="pcg"`` meets a backend that does
            not expose ``diag`` (propagated from ``cvx.linalg``).
    """
    op = _require_operator(a)
    _check_dimension(op, b)
    dinv: Vector | None = None  # Jacobi preconditioner, read off op.diag on first use

    def sub_solve(idx: NDArray[np.int_], x0: Vector | None) -> tuple[Vector, Vector | None, int]:
        """Solve the reduced system ``A_F x_F = b_F`` with the chosen inner solver."""
        nonlocal dinv
        if inner == "exact":
            rcond = op.rcond_free(idx)
            if rcond < _RCOND_MIN:
                msg = f"free block of size {idx.size} is numerically singular (rcond={rcond:.2e})"
                raise ValueError(msg)
            return op.solve_free(idx, b[idx]), None, 1
        if inner == "pcg":
            if dinv is None:
                dinv = 1.0 / op.diag
            xf, k_step = pcg(_free_matvec(op, idx), b[idx], dinv[idx], tol=cg_tol, maxit=cg_maxit)
            return xf, None, k_step
        xf, k_step = cg(_free_matvec(op, idx), b[idx], tol=cg_tol, maxit=cg_maxit, x0=x0)
        return xf, None, k_step

    def reduced_gradient(x: Vector, lam: Vector | None) -> Vector:  # noqa: ARG001
        """Return the reduced gradient ``s = A x - b``."""
        return op.matvec(x) - b

    return _active_set_loop(
        len(b),
        sub_solve,
        reduced_gradient,
        tol=tol,
        p_max=p_max,
        track=track,
        max_outer=max_outer,
        warm=warm,
    )


def solve_nnqp_eq(
    a: SymmetricOperator,
    b: Vector,
    b_eq: Matrix,
    c_eq: Vector,
    tol: float = 1e-8,
    cg_tol: float = 1e-10,
    p_max: int = 3,
    warm: tuple[NDArray[np.bool_], Vector] | None = None,
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
        warm: Optional ``(free_mask, x_prev)`` pair from a previous solve —
            the same tuple :func:`solve_nnqp` accepts. Starts the loop from
            that free set and seeds the ``v0`` solve of every saddle step
            from the newest iterate; the ``v1`` columns are re-solved cold.
            Across a support-stable parameter step the loop then terminates
            in a single outer step.

    Returns:
        A :class:`Result` with the multipliers in ``lam``. The reduced
        gradient underlying the dual test is ``s = A x - b - B^T lam``.

    Raises:
        TypeError: When ``a`` is not a :class:`cvx.linalg.SymmetricOperator`.
        ValueError: When the operator dimension does not match ``len(b)``.
    """
    op = _require_operator(a)
    _check_dimension(op, b)
    p = b_eq.shape[0]

    def sub_solve(idx: NDArray[np.int_], x0: Vector | None) -> tuple[Vector, Vector | None, int]:
        """Solve the saddle system on the free set via the p-by-p Schur complement."""
        matvec_f = _free_matvec(op, idx)
        b_f = b_eq[:, idx]
        v0, k0 = cg(matvec_f, b[idx], tol=cg_tol, x0=x0)
        v1 = np.zeros((idx.size, p))
        k_cols = 0
        for j in range(p):
            v1[:, j], kj = cg(matvec_f, b_f[j], tol=cg_tol)
            k_cols += kj
        schur = b_f @ v1  # p-by-p Schur complement, SPD
        lam = cholesky_solve(schur, c_eq - b_f @ v0)
        xf = v0 + v1 @ lam  # x_F = A_F^{-1}(b_F + B_F^T lambda)
        return xf, lam, k0 + k_cols

    def reduced_gradient(x: Vector, lam: Vector | None) -> Vector:
        """Return the constrained reduced gradient ``s = A x - b - B^T lam``."""
        correction = b_eq.T @ lam if lam is not None else np.zeros_like(b)
        return op.matvec(x) - b - correction

    return _active_set_loop(len(b), sub_solve, reduced_gradient, tol=tol, p_max=p_max, warm=warm)
