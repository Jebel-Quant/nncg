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

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from cvx.linalg import Matrix, SymmetricOperator, Vector
from numpy.typing import NDArray

from ._active_set import ReducedGradient, SubSolve, _drive
from ._equality import _saddle_solve
from .certificate import _require_operator


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
        _require_operator(a, b)

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
        _require_operator(a, b)

        def sub_solve(idx: NDArray[np.int_], x0: Vector | None) -> tuple[Vector, Vector | None, int]:
            """Solve the saddle system on the free set via the p-by-p Schur complement."""
            return _saddle_solve(self.inner, a, b, b_eq, c_eq, idx, x0)

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
        cfg = self.config
        x, outer, inner_total, fallback, converged, free, lam, traj = _drive(
            cfg.tol, cfg.p_max, cfg.track, cfg.max_outer, n, sub_solve, reduced_gradient, warm
        )
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
