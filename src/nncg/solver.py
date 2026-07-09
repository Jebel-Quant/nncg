"""Non-negative conjugate gradients: the active-set / block-principal-pivoting loop.

Solves the strictly convex non-negative quadratic program

    min_{x >= 0}  1/2 x^T A x - b^T x,        A symmetric positive definite,

and its equality-augmented variant with a general linear system ``B x = c``,
by wrapping a matrix-free inner solver in a primal-dual active-set outer loop.
The working-set toggles are the principal pivots of the linear complementarity
problem LCP(A, -b); guarding the fast block-pivot path with a least-index Bland
fallback gives unconditional finite termination at the unique global minimiser
— no non-degeneracy assumption (Theorem 5.1 of the accompanying paper). See
https://github.com/Jebel-Quant/mean_variance_solvers.

:class:`ActiveSetSolver` is the outer loop and the entry point. It knows nothing
about preconditioning: it asks its :class:`nncg.inner.InnerSolver` for a
per-free-block solve and drives the pivots around it. The quadratic term enters
as a :class:`cvx.linalg.SymmetricOperator`, accessed only through block products
— wrap an explicit SPD array in ``DenseOperator``, or pass ``GramOperator(M,
ridge)`` for ``A = M^T M + ridge I`` so the ``n x n`` matrix is never formed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from cvx.linalg import Matrix, SymmetricOperator, Vector, cholesky_solve
from numpy.typing import NDArray

SubSolve = Callable[[NDArray[np.int_], "Vector | None"], "tuple[Vector, Vector | None, int]"]
"""Subproblem solve on a free set: ``(idx, x0) -> (x_F, lam, inner_iters)``."""

ReducedGradient = Callable[["Vector", "Vector | None"], "Vector"]
"""Reduced gradient of the subproblem: ``(x, lam) -> s``."""


class InnerSolver(Protocol):
    """The inner-solver interface the active-set loop depends on (dependency inversion).

    Structural (a :class:`typing.Protocol`): anything with a matching
    :meth:`solve` is an inner solver, so implementations need neither import
    nor subclass this — this module (the high-level loop) owns the interface, and
    the implementations depend on it, not the other way round. The built-ins live
    in :mod:`nncg.inner` (:class:`~nncg.inner.CG`, :class:`~nncg.inner.Jacobi`,
    :class:`~nncg.inner.Nystrom`, :class:`~nncg.inner.Exact`); further ones —
    Clarabel- or KKT-equation-based — live in Jebel-Quant/mean_variance_solvers.
    """

    def solve(self, op: SymmetricOperator, idx: NDArray[np.int_], rhs: Vector, x0: Vector | None) -> tuple[Vector, int]:
        """Solve the free-block system ``A[F, F] y = rhs``, warm-started at ``x0``.

        Returns the free-block solution and the inner iteration count (each
        direct solve counts as one). Called once per outer step by the
        bound-constrained loop, and once per ``p + 1`` right-hand side per outer
        step by the equality-augmented loop.
        """
        ...


@dataclass(frozen=True)
class ActiveSetConfig:
    """Configuration of the active-set outer loop (:class:`ActiveSetSolver`).

    Bundles the outer-loop knobs into one argument; the inner solver and its
    tolerances live in :class:`nncg.inner.InnerSolver`, and the warm start stays
    a separate argument.

    Attributes:
        tol: Threshold of the primal and dual KKT violator tests.
        p_max: Patience budget — non-improving batch steps tolerated before a
            least-index Bland fallback pivot. Any value gives finite termination.
        track: Record the visited free-set trajectory in ``Result.traj``.
        max_outer: Optional cap on outer steps; when hit, the current iterate is
            returned with ``converged=False``.
    """

    tol: float = 1e-8
    p_max: int = 3
    track: bool = False
    max_outer: int | None = None


_NEEDS_OPERATOR = (
    "the quadratic term must be a cvx.linalg.SymmetricOperator: wrap a dense SPD "
    "array in DenseOperator(a), or pass GramOperator(M, ridge) for A = M'M + ridge*I"
)


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
    if not isinstance(a, SymmetricOperator):
        raise TypeError(_NEEDS_OPERATOR)
    if a.n != len(b):
        msg = f"operator dimension {a.n} does not match len(b) = {len(b)}"
        raise ValueError(msg)
    s = a.matvec(x) - b
    return float(
        max(
            np.max(-x, initial=0.0),
            np.max(-s, initial=0.0),
            np.max(np.abs(x * s), initial=0.0),
        )
    )


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


@dataclass(frozen=True)
class ActiveSetSolver:
    """The primal-dual active-set outer loop for the non-negative quadratic program.

    Holds the outer-loop :class:`ActiveSetConfig` and an
    :class:`nncg.inner.InnerSolver`, and drives the guarded block-pivot loop
    around the per-free-block solve the inner solver provides. It never touches a
    preconditioner — everything about CG/PCG/Nyström lives in ``inner``.

    Attributes:
        inner: The inner solver for each free block — e.g. :class:`nncg.inner.CG`
            (plain CG), :class:`nncg.inner.Jacobi`, :class:`nncg.inner.Nystrom`
            or :class:`nncg.inner.Exact`.
        config: Outer-loop configuration (violator tolerance, patience,
            trajectory tracking, outer-step cap).
    """

    inner: InnerSolver
    config: ActiveSetConfig = field(default_factory=ActiveSetConfig)

    def solve(
        self,
        a: SymmetricOperator,
        b: Vector,
        warm: tuple[NDArray[np.bool_], Vector] | None = None,
    ) -> Result:
        """Minimise ``1/2 x^T A x - b^T x`` over ``x >= 0`` by the active-set loop.

        Each free-block solve is delegated to :attr:`inner`; the reduced matrix
        is never materialised and ``A`` is never refactorised. The batch
        block-pivot fast path is guarded by a least-index Bland fallback, so
        termination at the unique global minimiser is unconditional.

        Args:
            a: The SPD operator ``A`` (a :class:`cvx.linalg.SymmetricOperator`) —
                ``DenseOperator`` for an explicit array, ``GramOperator(M, ridge)``
                for ``A = M^T M + ridge I`` whose Gram matrix is never formed.
            b: The linear term ``b``.
            warm: Optional ``(free_mask, x_prev)`` pair from a previous solve.
                Starts the loop from that free set and warm-starts every inner
                solve from the newest iterate (the :class:`nncg.inner.Exact`
                inner solver is direct, so it has nothing to seed but still
                starts from the warm free set) — across a support-stable
                parameter step the loop then terminates in a single outer step.

        Returns:
            A :class:`Result`; ``converged`` is True iff the KKT system was
            satisfied to ``config.tol``, which certifies the unique global
            minimiser.

        Raises:
            TypeError: When ``a`` is not a :class:`cvx.linalg.SymmetricOperator`.
            ValueError: When the operator dimension does not match ``len(b)``, or
                on the inner solver's own conditions in
                :meth:`InnerSolver.solve`.
            NotImplementedError: When a diagonal-preconditioned inner solver
                (:class:`nncg.inner.Jacobi`) meets a backend without ``diag``
                (propagated from ``cvx.linalg``).
        """
        if not isinstance(a, SymmetricOperator):
            raise TypeError(_NEEDS_OPERATOR)
        if a.n != len(b):
            msg = f"operator dimension {a.n} does not match len(b) = {len(b)}"
            raise ValueError(msg)

        def sub_solve(idx: NDArray[np.int_], x0: Vector | None) -> tuple[Vector, Vector | None, int]:
            """Solve the reduced system ``A_F x_F = b_F`` with the chosen inner solver."""
            xf, k_step = self.inner.solve(a, idx, b[idx], x0)
            return xf, None, k_step

        def reduced_gradient(x: Vector, lam: Vector | None) -> Vector:  # noqa: ARG001
            """Return the reduced gradient ``s = A x - b``."""
            return a.matvec(x) - b

        return self._run(len(b), sub_solve, reduced_gradient, warm)

    def solve_eq(
        self,
        a: SymmetricOperator,
        b: Vector,
        b_eq: Matrix,
        c_eq: Vector,
        warm: tuple[NDArray[np.bool_], Vector] | None = None,
    ) -> Result:
        """Solve ``min 1/2 x^T A x - b^T x`` subject to ``x >= 0`` and ``B x = c``.

        On each free set the saddle system is solved by eliminating the
        multiplier ``lambda`` in R^p through the p-by-p Schur complement
        ``S = B_F A_F^{-1} B_F^T``: the ``p + 1`` right-hand sides share the
        operator ``A_F`` and are each one inner solve, then ``S lambda = c - B_F
        v0`` fixes the multipliers in closed form. The single normalisation
        ``1^T x = beta`` is the ``p = 1`` case. ``B`` must have full row rank on
        the visited free sets (automatic for ``p = 1``).

        Args:
            a: The SPD operator ``A`` (a :class:`cvx.linalg.SymmetricOperator`).
            b: The linear term ``b``.
            b_eq: Equality matrix ``B`` of shape ``(p, n)``, full row rank.
            c_eq: Equality right-hand side ``c`` of shape ``(p,)``.
            warm: Optional ``(free_mask, x_prev)`` pair from a previous solve.
                Starts the loop from that free set and seeds the ``v0`` solve of
                every saddle step from the newest iterate; the ``v1`` columns are
                re-solved cold (their right-hand sides are the rows of ``B_F``,
                unrelated to ``x_prev``). Across a support-stable parameter step
                the loop then terminates in a single outer step.

        Returns:
            A :class:`Result` with the multipliers in ``lam``. The reduced
            gradient underlying the dual test is ``s = A x - b - B^T lam``.

        Raises:
            TypeError: When ``a`` is not a :class:`cvx.linalg.SymmetricOperator`.
            ValueError: When the operator dimension does not match ``len(b)``, or
                on the inner solver's own conditions in
                :meth:`InnerSolver.solve`.
            NotImplementedError: When a diagonal-preconditioned inner solver
                (:class:`nncg.inner.Jacobi`) meets a backend without ``diag``
                (propagated from ``cvx.linalg``).
        """
        if not isinstance(a, SymmetricOperator):
            raise TypeError(_NEEDS_OPERATOR)
        if a.n != len(b):
            msg = f"operator dimension {a.n} does not match len(b) = {len(b)}"
            raise ValueError(msg)
        p = b_eq.shape[0]

        def sub_solve(idx: NDArray[np.int_], x0: Vector | None) -> tuple[Vector, Vector | None, int]:
            """Solve the saddle system on the free set via the p-by-p Schur complement."""
            b_f = b_eq[:, idx]
            v0, k0 = self.inner.solve(a, idx, b[idx], x0)
            v1 = np.zeros((idx.size, p))
            k_cols = 0
            for j in range(p):
                v1[:, j], kj = self.inner.solve(a, idx, b_f[j], None)
                k_cols += kj
            schur = b_f @ v1  # p-by-p Schur complement, SPD
            lam = cholesky_solve(schur, c_eq - b_f @ v0)
            xf = v0 + v1 @ lam  # x_F = A_F^{-1}(b_F + B_F^T lambda)
            return xf, lam, k0 + k_cols

        def reduced_gradient(x: Vector, lam: Vector | None) -> Vector:
            """Return the constrained reduced gradient ``s = A x - b - B^T lam``."""
            correction = b_eq.T @ lam if lam is not None else np.zeros_like(b)
            return a.matvec(x) - b - correction

        return self._run(len(b), sub_solve, reduced_gradient, warm)

    def _run(
        self,
        n: int,
        sub_solve: SubSolve,
        reduced_gradient: ReducedGradient,
        warm: tuple[NDArray[np.bool_], Vector] | None,
    ) -> Result:
        """Run the guarded primal-dual active-set loop.

        The driver owns everything the termination proof depends on: the primal
        and dual violator tests, the batch exchange with its patience counter,
        and the least-index Bland fallback. What is solved on each free set — a
        single reduced system (:meth:`solve`), or the equality-augmented saddle
        system (:meth:`solve_eq`) — enters through the ``sub_solve`` callback,
        with ``reduced_gradient`` supplying the matching dual test quantity. The
        thresholds (``tol``, ``p_max``, ``track``, ``max_outer``) are read from
        :attr:`config`.

        Args:
            n: Problem dimension.
            sub_solve: Callback ``(idx, x0) -> (x_F, lam, inner_iters)`` solving
                the subproblem on the free set ``idx``. ``x0`` is a warm inner
                guess restricted to ``idx`` (None on a cold start); ``lam`` are
                the equality multipliers (None for the bound-only problem).
            reduced_gradient: Callback ``(x, lam) -> s`` computing the reduced
                gradient that drives the dual violator test.
            warm: Optional ``(free_mask, x_prev)`` pair from a previous solve.
                Starts the loop from that free set and seeds every subproblem
                solve from the newest iterate.

        Returns:
            A :class:`Result`; ``lam`` is whatever the last subproblem returned.
        """
        tol, p_max = self.config.tol, self.config.p_max
        free, x_guess, traj, cap = self._init_run_state(n, warm)
        x: Vector = np.zeros(n)
        lam: Vector | None = None
        n_bar, patience = n + 1, p_max
        outer = inner_total = fallback = 0
        converged = True

        while outer < cap:
            x, lam, k_step = self._solve_free_set(free, n, sub_solve, x_guess, traj)
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

    def _init_run_state(
        self, n: int, warm: tuple[NDArray[np.bool_], Vector] | None
    ) -> tuple[NDArray[np.bool_], Vector | None, list[tuple[int, ...]] | None, float]:
        """Seed the free set, inner guess, trajectory log and outer-iteration cap.

        Delegates the free set and warm inner guess to :func:`_init_active_set`,
        allocates the trajectory list only when ``config.track`` is set, and
        resolves ``config.max_outer`` (``None`` for uncapped) into a numeric
        loop bound so the driver's ``while`` condition stays branch-free.

        Returns:
            ``(free, x_guess, traj, cap)``: the initial free mask, the inner warm
            guess (``None`` on a cold start), the trajectory list (or ``None``),
            and the outer-iteration cap (``math.inf`` when uncapped).
        """
        free, x_guess = _init_active_set(n, warm)
        traj: list[tuple[int, ...]] | None = [] if self.config.track else None
        cap = math.inf if self.config.max_outer is None else self.config.max_outer
        return free, x_guess, traj, cap

    def _solve_free_set(
        self,
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
