"""One-call convenience entry points over the core solvers.

:func:`solve_nnqp` and :func:`solve_nnqp_eq` compose the three pieces of the
core API — wrap a plain SPD array in ``DenseOperator``, default-construct the
inner solver from a bare string, bundle the outer-loop knobs into an
:class:`~nncg.solver.ActiveSetConfig` — and delegate to
:class:`~nncg.solver.ActiveSetSolver`. They hold no logic of their own: reach
past them to ``ActiveSetSolver`` directly whenever you need to reuse a
configured solver across problems, or an inner solver the string shortcut cannot
express (``inner=Nystrom(nystrom=NystromConfig(rank=20))`` still works here,
passed as an instance). :func:`solve_nnqp_mprgp` is the matching one-call wrapper
over the projection-based :class:`~nncg.mprgp.MPRGP` solver for the same
bound-constrained problem.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from cvx.linalg import DenseOperator, Matrix, SymmetricOperator, Vector
from numpy.typing import NDArray

from .mprgp import MPRGP, MPRGPConfig, MPRGPResult
from .inner import CG, Exact, GlobalNystrom, Jacobi, Nystrom
from .solver import ActiveSetConfig, ActiveSetSolver, InnerSolver, Result

#: Bare-string shortcuts mapping to a default-constructed inner solver.
_INNER: dict[str, Callable[[], InnerSolver]] = {
    "cg": CG,
    "jacobi": Jacobi,
    "nystrom": Nystrom,
    "global_nystrom": GlobalNystrom,
    "exact": Exact,
}

InnerKind = Literal["cg", "jacobi", "nystrom", "global_nystrom", "exact"]
"""The bare-string shortcuts accepted for ``inner`` (keys of :data:`_INNER`)."""


def _resolve_inner(inner: InnerSolver | InnerKind) -> InnerSolver:
    """Return the inner solver, default-constructing it from a shortcut string.

    Args:
        inner: An :class:`~nncg.solver.InnerSolver` instance (returned as-is), or
            one of the shortcut strings in :data:`_INNER`.

    Returns:
        An inner-solver instance.

    Raises:
        ValueError: When ``inner`` is a string outside the shortcut set.
    """
    if isinstance(inner, str):
        try:
            return _INNER[inner]()
        except KeyError:
            valid = ", ".join(map(repr, _INNER))
            msg = f"unknown inner solver {inner!r}; pass an InnerSolver instance or one of {valid}"
            raise ValueError(msg) from None
    return inner


def _as_operator(a: SymmetricOperator | NDArray[np.float64]) -> SymmetricOperator:
    """Return ``a`` as a :class:`cvx.linalg.SymmetricOperator`, wrapping a plain array.

    A :class:`~cvx.linalg.SymmetricOperator` is used unchanged; anything else is
    treated as an explicit SPD array and wrapped in ``DenseOperator``. The
    matrix-free ``A = M^T M + ridge I`` path is deliberately *not* inferred from
    an ``M`` — pass ``GramOperator(M, ridge)`` explicitly for it.

    Args:
        a: A symmetric operator, or a 2-D SPD array.

    Returns:
        The operator form of ``a``.
    """
    if isinstance(a, SymmetricOperator):
        return a
    return DenseOperator(np.asarray(a, dtype=np.float64))


def solve_nnqp(
    a: SymmetricOperator | NDArray[np.float64],
    b: Vector,
    *,
    inner: InnerSolver | InnerKind = "cg",
    warm: tuple[NDArray[np.bool_], Vector] | None = None,
    tol: float = 1e-8,
    p_max: int = 3,
    track: bool = False,
    max_outer: int | None = None,
) -> Result:
    """Minimise ``1/2 x^T A x - b^T x`` over ``x >= 0`` — the one-call entry point.

    A thin convenience wrapper that composes the three pieces of the layered API
    for the common case: it wraps a plain SPD array in ``DenseOperator``, default-
    constructs the inner solver from a bare string, and bundles the outer-loop
    knobs into an :class:`ActiveSetConfig`, then delegates to
    :meth:`ActiveSetSolver.solve`. It holds no logic of its own — reach past it to
    :class:`ActiveSetSolver` directly whenever you need to reuse a configured
    solver across problems, or an inner solver this shortcut cannot express.

    Args:
        a: The SPD quadratic term. A :class:`cvx.linalg.SymmetricOperator` is used
            as-is; a plain 2-D array is wrapped in ``DenseOperator``. The matrix-
            free ``A = M^T M + ridge I`` path is *not* inferred from an ``M`` —
            pass ``GramOperator(M, ridge)`` explicitly for it.
        b: The linear term ``b``.
        inner: The inner solver for each free block, as an
            :class:`nncg.inner.InnerSolver` instance (fully configurable — e.g.
            ``Nystrom(nystrom=NystromConfig(rank=20))``), or one of the shortcut
            strings ``"cg"``, ``"jacobi"``, ``"nystrom"``, ``"global_nystrom"``,
            ``"exact"`` for its
            default configuration.
        warm: Optional ``(free_mask, x_prev)`` pair from a previous solve, forwarded
            to :meth:`ActiveSetSolver.solve` — see there for the warm-start semantics.
        tol: Threshold of the primal and dual KKT violator tests
            (``ActiveSetConfig.tol``).
        p_max: Patience budget before a least-index Bland fallback pivot
            (``ActiveSetConfig.p_max``).
        track: Record the visited free-set trajectory in ``Result.traj``.
        max_outer: Optional cap on outer steps; when hit, the current iterate is
            returned with ``converged=False``.

    Returns:
        A :class:`Result`; ``converged`` is True iff the KKT system was satisfied
        to ``tol``, which certifies the unique global minimiser.

    Raises:
        TypeError: When ``a`` is neither a :class:`cvx.linalg.SymmetricOperator`
            nor an array wrappable by ``DenseOperator``.
        ValueError: When ``inner`` is a string outside the shortcut set, when the
            operator dimension does not match ``len(b)``, or on the inner solver's
            own conditions.
    """
    config = ActiveSetConfig(tol=tol, p_max=p_max, track=track, max_outer=max_outer)
    solver = ActiveSetSolver(inner=_resolve_inner(inner), config=config)
    return solver.solve(_as_operator(a), b, warm=warm)


def solve_nnqp_eq(
    a: SymmetricOperator | NDArray[np.float64],
    b: Vector,
    b_eq: Matrix,
    c_eq: Vector,
    *,
    inner: InnerSolver | InnerKind = "cg",
    warm: tuple[NDArray[np.bool_], Vector] | None = None,
    tol: float = 1e-8,
    p_max: int = 3,
    track: bool = False,
    max_outer: int | None = None,
) -> Result:
    """Solve ``min 1/2 x^T A x - b^T x`` s.t. ``x >= 0`` and ``B x = c`` — one call.

    The equality-augmented companion to :func:`solve_nnqp`, with identical
    wrapping and configuration conventions; it delegates to
    :meth:`ActiveSetSolver.solve_eq`, where the per-free-set saddle system and the
    full-row-rank requirement on ``B`` are documented. The single normalisation
    ``1^T x = beta`` is the ``p = 1`` case.

    Args:
        a: The SPD quadratic term — a :class:`cvx.linalg.SymmetricOperator`, or a
            plain array wrapped in ``DenseOperator`` (see :func:`solve_nnqp`).
        b: The linear term ``b``.
        b_eq: Equality matrix ``B`` of shape ``(p, n)``, full row rank on the
            visited free sets.
        c_eq: Equality right-hand side ``c`` of shape ``(p,)``.
        inner: The inner solver instance, or a shortcut string — see
            :func:`solve_nnqp`.
        warm: Optional ``(free_mask, x_prev)`` pair, forwarded to
            :meth:`ActiveSetSolver.solve_eq`.
        tol: KKT violator tolerance (``ActiveSetConfig.tol``).
        p_max: Bland-fallback patience budget (``ActiveSetConfig.p_max``).
        track: Record the visited free-set trajectory in ``Result.traj``.
        max_outer: Optional outer-step cap; ``converged=False`` when hit.

    Returns:
        A :class:`Result` with the equality multipliers in ``lam``. The reduced
        gradient underlying the dual test is ``s = A x - b - B^T lam``.

    Raises:
        TypeError: When ``a`` is neither a :class:`cvx.linalg.SymmetricOperator`
            nor an array wrappable by ``DenseOperator``.
        ValueError: When ``inner`` is a string outside the shortcut set, when the
            operator dimension does not match ``len(b)``, or on the inner solver's
            own conditions.
    """
    config = ActiveSetConfig(tol=tol, p_max=p_max, track=track, max_outer=max_outer)
    solver = ActiveSetSolver(inner=_resolve_inner(inner), config=config)
    return solver.solve_eq(_as_operator(a), b, b_eq, c_eq, warm=warm)


def solve_nnqp_mprgp(
    a: SymmetricOperator | NDArray[np.float64],
    b: Vector,
    *,
    x0: Vector | None = None,
    tol: float = 1e-8,
    gamma: float = 1.0,
    alpha_bar: float | None = None,
    max_iter: int = 100_000,
    seed: int = 0,
) -> MPRGPResult:
    """Minimise ``1/2 x^T A x - b^T x`` over ``x >= 0`` by MPRGP — one call.

    The projection-based companion to :func:`solve_nnqp`: it solves the same
    bound-constrained program with Dostál & Schöberl's MPRGP
    (:class:`nncg.mprgp.MPRGP`) instead of the active-set loop — matrix-free and
    factorisation-free, so it never forms or refactorises ``A``. Like
    :func:`solve_nnqp` it wraps a plain SPD array in ``DenseOperator`` and bundles
    the knobs into an :class:`nncg.mprgp.MPRGPConfig`, then delegates to
    :meth:`nncg.mprgp.MPRGP.solve`. The equality-augmented variant is not covered
    — use :func:`solve_nnqp_eq` for ``B x = c``.

    Args:
        a: The SPD quadratic term. A :class:`cvx.linalg.SymmetricOperator` is used
            as-is; a plain 2-D array is wrapped in ``DenseOperator`` (the
            matrix-free ``A = M^T M + ridge I`` path is *not* inferred — pass
            ``GramOperator(M, ridge)`` explicitly for it).
        b: The linear term ``b``.
        x0: Optional feasible warm start, projected onto ``x >= 0``; ``None``
            starts from the origin.
        tol: Relative projected-gradient stopping tolerance
            (``MPRGPConfig.tol``).
        gamma: Proportioning constant ``Gamma > 0`` (``MPRGPConfig.gamma``).
        alpha_bar: Fixed projected-gradient step in ``(0, 2/||A||]``; ``None``
            estimates ``1/||A||`` matrix-free (``MPRGPConfig.alpha_bar``).
        max_iter: Iteration cap; ``converged=False`` when hit
            (``MPRGPConfig.max_iter``).
        seed: Seed of the power-iteration ``||A||`` estimate
            (``MPRGPConfig.seed``).

    Returns:
        An :class:`nncg.mprgp.MPRGPResult`; ``converged`` is True iff the
        projected gradient fell below ``tol * ||b||``, which certifies the unique
        global minimiser.

    Raises:
        TypeError: When ``a`` is neither a :class:`cvx.linalg.SymmetricOperator`
            nor an array wrappable by ``DenseOperator``.
        ValueError: When the operator dimension does not match ``len(b)``, when
            ``gamma`` is not strictly positive, or when ``alpha_bar`` is set but
            not strictly positive.
    """
    config = MPRGPConfig(tol=tol, gamma=gamma, alpha_bar=alpha_bar, max_iter=max_iter, seed=seed)
    return MPRGP(config=config).solve(_as_operator(a), b, x0=x0)
