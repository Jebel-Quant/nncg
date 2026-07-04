"""Alternative solvers for the non-negative quadratic program — benchmark baselines.

Reference implementations that solve the same problems as :mod:`nncg.solver`,
kept here (outside the installed package, next to ``tests/problems.py``) so the
core library stays dependency-pure — NumPy and ``cvx-linalg`` only — while the
numerical study can still line ``nncg`` up against established solvers.

All five solve ``min_{x>=0} 1/2 x^T A x - b^T x`` (the bound-only NNQP), and the
first two also handle the equality-augmented variant ``B x = c``:

- :func:`solve_osqp` — the OSQP operator-splitting (ADMM) QP solver.
- :func:`solve_clarabel` — the Clarabel interior-point conic solver.
- :func:`solve_lawson_hanson` — the classical Lawson & Hanson (1974) active-set
  NNLS algorithm, with each unconstrained passive-set solve done by the
  in-house CG of :mod:`nncg.krylov` (hence "Lawson-Hanson (cg)").
- :func:`solve_fista` — Beck & Teboulle's (2009) accelerated proximal-gradient
  method (FISTA), the prox being the projection onto the non-negative orthant.
- :func:`solve_duchi` — the same FISTA core with the prox replaced by Duchi et
  al.'s (2008) exact projection onto the simplex ``{x >= 0, 1^T x = beta}``.
  Defined **only** for the ``p = 1`` all-ones normalisation constraint.

Each returns a :class:`BaselineResult` carrying the minimiser, an iteration
count, wall-clock time and a status string, so a harness can tabulate them
against :class:`nncg.solver.Result`. OSQP, Clarabel and SciPy are imported
lazily inside the solvers that need them, so this module (and the pure-NumPy
Lawson-Hanson / Duchi routines) imports without them installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from cvx.linalg import Matrix, SymmetricOperator, Vector
from numpy.typing import NDArray

from nncg.krylov import cg

Prox = Callable[[Vector], Vector]
"""A proximal / projection step ``v -> prox(v)`` of a FISTA iteration."""


@dataclass(frozen=True)
class BaselineResult:
    """Outcome of an alternative solver, shaped for comparison with ``Result``.

    Attributes:
        x: The computed minimiser.
        iters: Iteration count in the solver's native unit — outer ADMM /
            interior-point iterations for OSQP / Clarabel, total inner CG
            iterations for Lawson-Hanson, projected-gradient steps for Duchi.
        time_s: Wall-clock solve time in seconds (setup plus solve).
        status: Solver-reported status string (e.g. ``"solved"``).
        lam: Equality multipliers when the problem carried a ``B x = c``
            constraint; None for the bound-only problem.
    """

    x: Vector
    iters: int
    time_s: float
    status: str
    lam: Vector | None = None


def _densify(a: SymmetricOperator) -> Matrix:
    """Materialise the dense symmetric matrix of an operator.

    The baselines need an explicit ``A`` (OSQP/Clarabel want sparse CSC, the
    active-set and first-order routines slice and multiply it directly), so the
    matrix-free contract of :mod:`nncg` is deliberately given up here. The
    matrix is rebuilt column by column through ``matvec`` and symmetrised.

    Args:
        a: The SPD operator ``A``.

    Returns:
        The dense ``n x n`` array of ``A``.
    """
    n = a.n
    cols = np.column_stack([a.matvec(e) for e in np.eye(n)])
    return 0.5 * (cols + cols.T)


def solve_osqp(
    a: SymmetricOperator,
    b: Vector,
    b_eq: Matrix | None = None,
    c_eq: Vector | None = None,
    tol: float = 1e-9,
    max_iter: int = 40_000,
) -> BaselineResult:
    """Solve the NNQP with OSQP, the operator-splitting (ADMM) QP solver.

    OSQP minimises ``1/2 x^T P x + q^T x`` subject to ``l <= A_c x <= u``. The
    quadratic ``P`` is ``A`` (upper triangle, sparse) and ``q = -b``; the bound
    ``x >= 0`` is the row block ``I x >= 0`` and each equality row of ``B``
    enters as ``l = u = c``. Polishing is enabled so the ADMM iterate is
    refined to a high-accuracy active-set solution.

    Args:
        a: The SPD operator ``A``.
        b: The linear term ``b``.
        b_eq: Optional equality matrix ``B`` of shape ``(p, n)``.
        c_eq: Optional equality right-hand side ``c`` of shape ``(p,)``.
        tol: Absolute and relative stopping tolerance (``eps_abs = eps_rel``).
        max_iter: Iteration cap.

    Returns:
        A :class:`BaselineResult`; ``iters`` is the OSQP iteration count and
        ``lam`` the equality multipliers when ``b_eq`` was given.
    """
    import osqp
    from scipy import sparse

    mat = _densify(a)
    n = a.n
    p_mat = sparse.triu(sparse.csc_matrix(mat)).tocsc()
    q = -np.asarray(b, dtype=float)

    blocks: list[Any] = [sparse.eye(n, format="csc")]
    lo: list[Vector] = [np.zeros(n)]
    hi: list[Vector] = [np.full(n, np.inf)]
    if b_eq is not None and c_eq is not None:
        blocks.append(sparse.csc_matrix(np.asarray(b_eq, dtype=float)))
        lo.append(np.asarray(c_eq, dtype=float))
        hi.append(np.asarray(c_eq, dtype=float))
    a_con = sparse.vstack(blocks, format="csc")
    lower, upper = np.concatenate(lo), np.concatenate(hi)

    t0 = perf_counter()
    prob = osqp.OSQP()
    prob.setup(
        P=p_mat,
        q=q,
        A=a_con,
        l=lower,
        u=upper,
        eps_abs=tol,
        eps_rel=tol,
        max_iter=max_iter,
        polishing=True,
        verbose=False,
    )
    res = prob.solve()
    elapsed = perf_counter() - t0

    x = np.asarray(res.x, dtype=float)
    lam = None
    if b_eq is not None:
        # dual of the equality rows (tail of y past the n bound rows), sign
        # flipped to nncg's s = A x - b - B^T lam convention
        lam = -np.asarray(res.y, dtype=float)[n:]
    return BaselineResult(x=x, iters=int(res.info.iter), time_s=elapsed, status=str(res.info.status), lam=lam)


def solve_clarabel(
    a: SymmetricOperator,
    b: Vector,
    b_eq: Matrix | None = None,
    c_eq: Vector | None = None,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> BaselineResult:
    """Solve the NNQP with Clarabel, the interior-point conic solver.

    Clarabel minimises ``1/2 x^T P x + q^T x`` subject to ``A_c x + s = h``,
    ``s in K``. The bound ``x >= 0`` is written ``-I x + s = 0``, ``s`` in the
    non-negative cone; equality rows ``B x = c`` are placed first in the
    zero cone (``B x + s = c``, ``s = 0``). ``P`` is the upper triangle of
    ``A`` and ``q = -b``.

    Args:
        a: The SPD operator ``A``.
        b: The linear term ``b``.
        b_eq: Optional equality matrix ``B`` of shape ``(p, n)``.
        c_eq: Optional equality right-hand side ``c`` of shape ``(p,)``.
        tol: Absolute and relative gap/feasibility tolerance.
        max_iter: Interior-point iteration cap.

    Returns:
        A :class:`BaselineResult`; ``iters`` is the interior-point iteration
        count and ``lam`` the equality multipliers when ``b_eq`` was given.
    """
    import clarabel
    from scipy import sparse

    mat = _densify(a)
    n = a.n
    p_mat = sparse.triu(sparse.csc_matrix(mat)).tocsc()
    q = -np.asarray(b, dtype=float)

    g_blocks: list[Any] = []
    h_blocks: list[Vector] = []
    cones: list[Any] = []
    p = 0
    if b_eq is not None and c_eq is not None:
        p = np.asarray(b_eq).shape[0]
        g_blocks.append(sparse.csc_matrix(np.asarray(b_eq, dtype=float)))
        h_blocks.append(np.asarray(c_eq, dtype=float))
        cones.append(clarabel.ZeroConeT(p))
    g_blocks.append(sparse.csc_matrix(-np.eye(n)))
    h_blocks.append(np.zeros(n))
    cones.append(clarabel.NonnegativeConeT(n))
    g_con = sparse.vstack(g_blocks, format="csc")
    h_con = np.concatenate(h_blocks)

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = tol
    settings.tol_gap_rel = tol
    settings.tol_feas = tol
    settings.max_iter = max_iter

    t0 = perf_counter()
    solver = clarabel.DefaultSolver(p_mat, q, g_con, h_con, cones, settings)
    sol = solver.solve()
    elapsed = perf_counter() - t0

    x = np.asarray(sol.x, dtype=float)
    # z holds the conic duals in row order; the equality multiplier is the
    # zero-cone block (sign flipped to match s = A x - b - B^T lam).
    lam = -np.asarray(sol.z, dtype=float)[:p] if p else None
    return BaselineResult(x=x, iters=int(sol.iterations), time_s=elapsed, status=str(sol.status), lam=lam)


def solve_lawson_hanson(
    a: SymmetricOperator,
    b: Vector,
    tol: float = 1e-9,
    cg_tol: float = 1e-11,
    max_outer: int | None = None,
) -> BaselineResult:
    """Solve the bound-only NNQP by Lawson & Hanson's active-set NNLS, CG inner.

    The classical NNLS active-set method (Lawson & Hanson 1974, Algorithm
    NNLS) applied to ``min_{x>=0} 1/2 x^T A x - b^T x``: it grows a passive
    (free) set one index at a time by the most-violated dual ``w = b - A x``,
    solving the unconstrained reduced system ``A_P x_P = b_P`` on the passive
    set, and takes a ratio-test step back whenever that solve leaves the
    non-negative orthant. Unlike the block-pivot loop in :mod:`nncg`, it
    exchanges a single index per step. Each reduced solve is the in-house CG of
    :mod:`nncg.krylov`, so the reported ``iters`` is directly comparable to
    ``Result.inner``.

    Args:
        a: The SPD operator ``A``.
        b: The linear term ``b``.
        tol: Threshold of the dual violator test ``max_j w_j > tol``.
        cg_tol: Relative residual tolerance of the inner CG solves.
        max_outer: Optional cap on outer steps; defaults to ``3 * n``.

    Returns:
        A :class:`BaselineResult`; ``iters`` is the total inner CG iterations
        and ``status`` is ``"solved"`` or ``"max_outer"``.
    """
    mat = _densify(a)
    n = a.n
    rhs = np.asarray(b, dtype=float)
    cap = 3 * n if max_outer is None else max_outer

    passive = np.zeros(n, dtype=bool)  # P: free indices
    x = np.zeros(n)
    inner = 0
    status = "solved"

    t0 = perf_counter()
    for _ in range(cap):
        w = rhs - mat @ x  # dual / negative gradient
        cand = np.flatnonzero(~passive)
        if cand.size == 0 or float(np.max(w[cand])) <= tol:
            break
        passive[cand[int(np.argmax(w[cand]))]] = True  # add most-violated index

        while True:
            idx = np.flatnonzero(passive)
            z_p, k = cg(lambda v, i=idx: mat[np.ix_(i, i)] @ v, rhs[idx], tol=cg_tol)
            inner += k
            if z_p.size == 0 or float(np.min(z_p)) > 0.0:
                x = np.zeros(n)
                x[idx] = z_p
                break
            # ratio test: back off to the first bound the full step would cross
            bad = z_p <= 0.0
            ratios = x[idx][bad] / (x[idx][bad] - z_p[bad])
            alpha = float(np.min(ratios))
            step = np.zeros(n)
            step[idx] = z_p - x[idx]
            x = x + alpha * step
            passive[idx[np.abs(x[idx]) <= tol]] = False  # drop the newly-bound indices
            x[~passive] = 0.0
    else:
        status = "max_outer"
    elapsed = perf_counter() - t0

    return BaselineResult(x=x, iters=inner, time_s=elapsed, status=status)


def _fista(
    mat: Matrix,
    rhs: Vector,
    prox: Prox,
    x0: Vector,
    step: float,
    tol: float,
    max_iter: int,
) -> tuple[Vector, int, str]:
    """The FISTA accelerated proximal-gradient core shared by the first-order solvers.

    Beck & Teboulle's (2009) FISTA on the smooth part ``1/2 x^T A x - b^T x``
    (gradient ``A x - b``, step ``1/L``) with the constraint handled by the
    ``prox`` step — projection onto the non-negative orthant for
    :func:`solve_fista`, onto the simplex for :func:`solve_duchi`. Nesterov
    momentum lifts the rate from ``O(kappa)`` to ``O(sqrt(kappa))`` iterations.

    Args:
        mat: The dense SPD matrix ``A``.
        rhs: The linear term ``b``.
        prox: The projection ``v -> prox(v)`` onto the feasible set.
        x0: Starting point (projected before the first step).
        step: Fixed step size (``1 / lambda_max(A)`` for the SPD gradient).
        tol: Relative step-size stopping tolerance ``||x_{k+1}-x_k|| <= tol``.
        max_iter: Iteration cap.

    Returns:
        The tuple ``(x, iters, status)``; ``status`` is ``"solved"`` on the
        step-size exit or ``"max_iter"`` when the cap is hit.
    """
    x = prox(x0)
    y = x.copy()
    t_mom = 1.0
    it = 0
    status = "max_iter"
    while it < max_iter:
        it += 1
        x_new = prox(y - step * (mat @ y - rhs))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom))
        y = x_new + ((t_mom - 1.0) / t_new) * (x_new - x)
        moved = float(np.linalg.norm(x_new - x))
        x, t_mom = x_new, t_new
        if moved <= tol * max(1.0, float(np.linalg.norm(x))):
            status = "solved"
            break
    return x, it, status


def solve_fista(
    a: SymmetricOperator,
    b: Vector,
    tol: float = 1e-8,
    max_iter: int = 200_000,
    step: float | None = None,
) -> BaselineResult:
    """Solve the bound-only NNQP by FISTA, projecting onto the non-negative orthant.

    The general first-order baseline for ``min_{x>=0} 1/2 x^T A x - b^T x``:
    accelerated proximal gradient (:func:`_fista`) whose prox is the elementwise
    clip ``max(x, 0)`` — the Euclidean projection onto ``{x >= 0}``. Unlike
    :func:`solve_duchi` it carries no equality constraint; unlike
    :func:`solve_lawson_hanson` it never solves a linear system, only matrix
    products, so it is the reference for the regime where ``A`` is too large to
    factor.

    Args:
        a: The SPD operator ``A``.
        b: The linear term ``b``.
        tol: Relative step-size stopping tolerance ``||x_{k+1}-x_k|| <= tol``.
        max_iter: Iteration cap.
        step: Fixed step size; defaults to ``1 / lambda_max(A)``.

    Returns:
        A :class:`BaselineResult`; ``iters`` is the projected-gradient step
        count. ``lam`` is left None (there is no equality constraint).
    """
    mat = _densify(a)
    rhs = np.asarray(b, dtype=float)
    if step is None:
        step = 1.0 / float(np.linalg.eigvalsh(mat)[-1])

    t0 = perf_counter()
    x, it, status = _fista(mat, rhs, lambda v: np.maximum(v, 0.0), np.zeros(a.n), step, tol, max_iter)
    elapsed = perf_counter() - t0
    return BaselineResult(x=x, iters=it, time_s=elapsed, status=status)


def _project_simplex(v: Vector, beta: float) -> Vector:
    """Euclidean projection onto the scaled simplex ``{x >= 0, 1^T x = beta}``.

    The exact ``O(n log n)`` sort-based projection of Duchi et al. (2008,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions").

    Args:
        v: The point to project.
        beta: The simplex scale (target of ``1^T x``).

    Returns:
        ``argmin_{x >= 0, 1^T x = beta} ||x - v||_2``.
    """
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - beta
    rho = int(np.nonzero(u - css / np.arange(1, v.size + 1) > 0)[0][-1])
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def solve_duchi(
    a: SymmetricOperator,
    b: Vector,
    beta: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 200_000,
    step: float | None = None,
) -> BaselineResult:
    """Solve the simplex-constrained NNQP by accelerated projected gradient.

    The simplex-constrained sibling of :func:`solve_fista`: the same FISTA core
    (:func:`_fista`) over ``min 1/2 x^T A x - b^T x`` on the simplex
    ``{x >= 0, 1^T x = beta}`` — the ``p = 1`` all-ones normalisation, and the
    **only** equality this routine handles — with the prox being the exact
    Duchi projection :func:`_project_simplex` instead of the orthant clip. The
    step is ``1/L`` (``L`` the largest eigenvalue of ``A``) and Nesterov
    momentum gives the ``O(sqrt(kappa))`` rate.

    Args:
        a: The SPD operator ``A``.
        b: The linear term ``b``.
        beta: Target of the normalisation ``1^T x = beta``.
        tol: Relative step-size stopping tolerance ``||x_{k+1}-x_k|| <= tol``.
        max_iter: Iteration cap.
        step: Fixed step size; defaults to ``1 / lambda_max(A)``.

    Returns:
        A :class:`BaselineResult`; ``iters`` is the projected-gradient step
        count. ``lam`` is left None (the multiplier is not formed).
    """
    mat = _densify(a)
    n = a.n
    rhs = np.asarray(b, dtype=float)
    if step is None:
        step = 1.0 / float(np.linalg.eigvalsh(mat)[-1])

    t0 = perf_counter()
    x, it, status = _fista(mat, rhs, lambda v: _project_simplex(v, beta), np.full(n, beta / n), step, tol, max_iter)
    elapsed = perf_counter() - t0
    return BaselineResult(x=x, iters=it, time_s=elapsed, status=status)


def ones_row(n: int) -> NDArray[np.float64]:
    """Return the ``1 x n`` all-ones equality matrix ``B = 1^T``.

    The equality ``B x = c`` that turns :func:`nncg.solver.solve_nnqp_eq`,
    :func:`solve_osqp` and :func:`solve_clarabel` into simplex solves matching
    :func:`solve_duchi`.

    Args:
        n: Problem dimension.

    Returns:
        A ``(1, n)`` array of ones.
    """
    return np.ones((1, n))
