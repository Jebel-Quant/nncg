"""Primal-dual active-set primitives for the driver loop in :mod:`nncg.solver`.

The pure working-set algebra the outer loop is built from, factored out of the
:class:`~nncg.solver.ActiveSetSolver` orchestration: seeding the free set from a
cold or warm start (:func:`_init_active_set`), splitting the KKT violators into
primal and dual sets (:func:`_violators`), applying one guarded working-set pivot
(:func:`_pivot`), and the per-outer-step scaffolding (:func:`_init_run_state`,
:func:`_solve_free_set`). None of it knows about preconditioning or the inner
solver — it operates on boolean free masks and index arrays alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from cvx.linalg import Vector
from numpy.typing import NDArray

SubSolve = Callable[[NDArray[np.int_], "Vector | None"], "tuple[Vector, Vector | None, int]"]
"""Subproblem solve on a free set: ``(idx, x0) -> (x_F, lam, inner_iters)``."""

ReducedGradient = Callable[["Vector", "Vector | None"], "Vector"]
"""Reduced gradient of the subproblem: ``(x, lam) -> s``."""


def _init_active_set(n: int, warm: tuple[NDArray[np.bool_], Vector] | None) -> tuple[NDArray[np.bool_], Vector | None]:
    """Seed the free set and inner warm guess for the outer loop.

    Cold (``warm is None``): everything free (``F = {1..n}``) with no inner
    guess. Warm: a copy of the previous free mask and the previous iterate as
    the seed for every subproblem solve.
    """
    if warm is None:
        return np.ones(n, dtype=bool), None  # F = {1..n} initially
    return warm[0].copy(), warm[1]


def _violators(
    free: NDArray[np.bool_], x: Vector, s: Vector, tol: float
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """Split the KKT violators at tolerance ``tol`` into primal and dual sets.

    Returns ``(prim, dual, viol)``: ``prim`` (set ``D``) are free indices whose
    primal value went negative, ``dual`` (set ``V``) are bound indices whose
    reduced gradient went negative, and ``viol`` is their concatenation. An empty
    ``viol`` certifies the KKT conditions at the unique global minimiser.
    """
    prim = np.flatnonzero(free & (x < -tol))  # D: free but negative
    dual = np.flatnonzero((~free) & (s < -tol))  # V: bound but s < 0
    return prim, dual, np.concatenate([prim, dual])


def _pivot(
    free: NDArray[np.bool_],
    prim: NDArray[np.int_],
    dual: NDArray[np.int_],
    viol: NDArray[np.int_],
    n_bar: int,
    patience: int,
    p_max: int,
) -> tuple[int, int, int]:
    """Apply one working-set pivot, mutating ``free`` in place.

    Takes the fast batch exchange — drop every primal violator ``D``, add every
    dual violator ``V`` — while the violator count strictly drops below ``n_bar``
    (patience reset to ``p_max``) or patience remains (decremented). Once patience
    is exhausted without progress it falls back to a single least-index Bland
    pivot, the load-bearing anti-cycling guarantee behind finite termination.

    Returns the updated ``(n_bar, patience, fallback_increment)``; the last is 1
    when the Bland fallback fired, 0 on the batch fast path.
    """
    n_viol = viol.size
    if n_viol < n_bar or patience > 0:  # fast path: progress, or patience remains
        if n_viol < n_bar:
            n_bar = n_viol
            patience = p_max
        else:
            patience -= 1
        free[prim] = False  # batch exchange: drop all D, add all V
        free[dual] = True
        return n_bar, patience, 0
    i_star = int(np.min(viol))  # anti-cycling fallback: single Bland least-index pivot
    free[i_star] = not free[i_star]
    return n_bar, patience, 1


def _init_run_state(
    n: int, track: bool, max_outer: int | None, warm: tuple[NDArray[np.bool_], Vector] | None
) -> tuple[NDArray[np.bool_], Vector | None, list[tuple[int, ...]] | None, float]:
    """Seed the free set, inner guess, trajectory log and outer-iteration cap.

    Delegates the free set and warm inner guess to :func:`_init_active_set`,
    allocates the trajectory list only when ``track`` is set, and resolves
    ``max_outer`` (``None`` for uncapped) into a numeric loop bound so the
    driver's ``while`` condition stays branch-free.

    Returns:
        ``(free, x_guess, traj, cap)``: the initial free mask, the inner warm
        guess (``None`` on a cold start), the trajectory list (or ``None``),
        and the outer-iteration cap (``math.inf`` when uncapped).
    """
    free, x_guess = _init_active_set(n, warm)
    traj: list[tuple[int, ...]] | None = [] if track else None
    cap = math.inf if max_outer is None else max_outer
    return free, x_guess, traj, cap


def _solve_free_set(
    free: NDArray[np.bool_],
    n: int,
    sub_solve: SubSolve,
    x_guess: Vector | None,
    traj: list[tuple[int, ...]] | None,
) -> tuple[Vector, Vector | None, int]:
    """Solve the subproblem on the current free set and scatter it into ``R^n``.

    Records the free set on ``traj`` when tracking, seeds the inner solve
    from ``x_guess`` restricted to the free set (cold when ``None``), and
    places the returned free-block solution back into a full zero vector.

    Args:
        free: Boolean free-set mask over the ``n`` variables.
        n: Problem dimension.
        sub_solve: Subproblem callback ``(idx, x0) -> (x_F, lam, inner_iters)``.
        x_guess: Inner warm guess over all variables, or ``None`` (cold).
        traj: Trajectory list to append the free set to, or ``None``.

    Returns:
        ``(x, lam, inner_iters)``: the full-length iterate, the equality
        multipliers (``None`` for the bound-only problem), and the inner
        iteration count from the callback.
    """
    idx = np.flatnonzero(free)
    if traj is not None:
        traj.append(tuple(idx.tolist()))
    x0 = x_guess[idx] if x_guess is not None else None
    xf, lam, k_step = sub_solve(idx, x0)
    x: Vector = np.zeros(n)
    x[idx] = xf
    return x, lam, k_step


def _drive(
    tol: float,
    p_max: int,
    track: bool,
    max_outer: int | None,
    n: int,
    sub_solve: SubSolve,
    reduced_gradient: ReducedGradient,
    warm: tuple[NDArray[np.bool_], Vector] | None,
) -> tuple[Vector, int, int, int, bool, NDArray[np.bool_], Vector | None, list[tuple[int, ...]] | None]:
    """Run the guarded primal-dual active-set loop and return its raw outcome.

    Owns everything the termination proof depends on: the primal and dual
    violator tests (:func:`_violators`), the batch exchange with its patience
    counter and the least-index Bland fallback (:func:`_pivot`). What is solved
    on each free set enters through ``sub_solve``, with ``reduced_gradient``
    supplying the matching dual test quantity; the thresholds come straight from
    :class:`nncg.solver.ActiveSetConfig`. Returns a plain tuple so this module
    need not depend on :class:`nncg.solver.Result` — the caller wraps it.

    Args:
        tol: Violator tolerance of the primal and dual KKT tests.
        p_max: Patience budget before the least-index Bland fallback pivot.
        track: Record the visited free-set trajectory.
        max_outer: Optional cap on outer steps (``None`` for uncapped).
        n: Problem dimension.
        sub_solve: Callback ``(idx, x0) -> (x_F, lam, inner_iters)`` solving the
            subproblem on the free set ``idx``.
        reduced_gradient: Callback ``(x, lam) -> s`` for the dual violator test.
        warm: Optional ``(free_mask, x_prev)`` pair from a previous solve.

    Returns:
        ``(x, outer, inner_total, fallback, converged, free, lam, traj)``.
    """
    free, x_guess, traj, cap = _init_run_state(n, track, max_outer, warm)
    x: Vector = np.zeros(n)
    lam: Vector | None = None
    n_bar, patience = n + 1, p_max
    outer = inner_total = fallback = 0
    converged = True

    while outer < cap:
        x, lam, k_step = _solve_free_set(free, n, sub_solve, x_guess, traj)
        outer += 1
        inner_total += k_step
        if x_guess is not None:
            x_guess = x  # warm mode: newest iterate seeds the next reduced solve
        prim, dual, viol = _violators(free, x, reduced_gradient(x, lam), tol)
        if viol.size == 0:
            break  # KKT satisfied -> unique global minimiser
        n_bar, patience, fired = _pivot(free, prim, dual, viol, n_bar, patience, p_max)
        fallback += fired
    else:
        converged = False  # outer cap reached without certifying KKT

    return x, outer, inner_total, fallback, converged, free, lam, traj
