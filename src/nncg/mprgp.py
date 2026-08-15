"""MPRGP: modified proportioning with reduced gradient projections.

A matrix-free, projection-based alternative outer solver for the same
strictly convex non-negative quadratic program the active-set loop targets,

    min_{x >= 0}  1/2 x^T A x - b^T x,        A symmetric positive definite.

Where :class:`nncg.solver.ActiveSetSolver` toggles a working set and solves an
unconstrained system on each free block, MPRGP (Dostál & Schöberl, 2005) never
factorises anything: it interleaves three cheap first-order moves, each costing
one or two Hessian products,

* a **conjugate-gradient step** that minimises within the current face while it
  stays feasible (the free set unchanged),
* an **expansion step** that walks to the nearest bound and takes one fixed-step
  projected-gradient move to *add* constraints to the active set, and
* a **proportioning step** along the chopped gradient that *removes* constraints
  from the active set,

switched by the proportioning test ``||beta(x)||^2 <= gamma^2 phi~(x)^T phi(x)``.
With the projected-gradient step bounded by ``alpha_bar in (0, 2/||A||]`` the
iteration converges for any feasible start, and — because it identifies the
active set of the minimiser in finitely many steps and then reduces to plain CG
on the optimal face — it terminates finitely in exact arithmetic. ``A`` enters
only through :meth:`cvx.linalg.SymmetricOperator.matvec`, so the ``n x n`` matrix
is never formed; ``||A||`` for the step bound is estimated matrix-free by power
iteration.

This is the bound-constrained solver; the equality-augmented variant ``B x = c``
is out of scope here (it needs an augmented-Lagrangian outer wrap, SMALBE/SMALSE
around MPRGP) — use :meth:`nncg.solver.ActiveSetSolver.solve_eq` for that.

Reference: Z. Dostál and J. Schöberl, "Minimizing quadratic functions subject to
bound constraints with the rate of convergence and finite termination",
Comput. Optim. Appl. 30 (2005), 23-43.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from cvx.linalg import SymmetricOperator, Vector, power_iteration
from numpy.typing import NDArray

from .certificate import _require_operator

MatVec = Callable[[Vector], Vector]
"""The action ``v -> A v`` of the SPD operator — MPRGP's only access to ``A``."""

Iterate = tuple[Vector, Vector, Vector]
"""The MPRGP iteration state ``(x, g, p)``: iterate, gradient ``A x - b``, CG direction.

Every move consumes one state and returns the next, so the loop in :func:`_mprgp`
carries no other mutable numerics — only the counters the result reports.
"""


def _free_gradient(x: Vector, g: Vector) -> Vector:
    """Free gradient ``phi``: the gradient on the free set, zero on the active set.

    ``phi_i = g_i`` where ``x_i > 0`` (free) and ``0`` where ``x_i = 0`` (active).
    It drives the conjugate-gradient and expansion steps, which move only the
    free variables.
    """
    return np.where(x > 0.0, g, 0.0)


def _chopped_gradient(x: Vector, g: Vector) -> Vector:
    """Chopped gradient ``beta``: the releasing part of the gradient on the active set.

    ``beta_i = min(g_i, 0)`` where ``x_i = 0`` (active) and ``0`` where
    ``x_i > 0`` (free). A negative gradient at a bound means the objective still
    decreases along the feasible ``+e_i`` direction, so the proportioning step
    follows ``beta`` to release such constraints.
    """
    return np.where(x > 0.0, 0.0, np.minimum(g, 0.0))


def _reduced_free_gradient(x: Vector, g: Vector, alpha_bar: float) -> Vector:
    """Reduced free gradient ``phi~``: the free gradient capped by the feasible step.

    ``phi~_i = min(g_i, x_i / alpha_bar)`` on the free set and ``0`` on the active
    set. It measures the decrease a single ``alpha_bar`` projected-gradient step
    can realise on each free variable (a downhill move is limited by the distance
    ``x_i`` to the bound), and enters only the proportioning test.
    """
    return np.where(x > 0.0, np.minimum(g, x / alpha_bar), 0.0)


def _max_feasible_step(x: Vector, p: Vector) -> float:
    """Largest ``alpha`` with ``x - alpha p >= 0``: ``min_{p_i > 0} x_i / p_i``.

    Only components that decrease (``p_i > 0``) can reach the bound; when none do,
    the whole ray is feasible and the step is unbounded (``+inf``).
    """
    decreasing = p > 0.0
    if not decreasing.any():
        return np.inf
    return float(np.min(x[decreasing] / p[decreasing]))


def _initial_iterate(matvec: MatVec, b: Vector, x0: Vector | None) -> Iterate:
    """Project the warm start onto ``x >= 0`` and seed the gradient and CG direction.

    Args:
        matvec: The action ``v -> A v`` of the SPD operator.
        b: The linear term ``b``.
        x0: Optional warm start, projected onto the feasible set; ``None`` starts
            at the origin.

    Returns:
        The initial ``(x, g, p)``. Costs the one Hessian product of ``g = A x - b``.
    """
    x = np.zeros_like(b) if x0 is None else np.maximum(np.asarray(x0, dtype=np.float64), 0.0)
    g = matvec(x) - b
    return x, g, _free_gradient(x, g)


def _stopping_threshold(b: Vector, tol: float) -> float:
    """Absolute exit threshold ``tol * ||b||`` of the projected-gradient stopping test.

    ``||b||`` is replaced by ``1`` when ``b = 0``, so the test stays meaningful on
    the degenerate problem whose minimiser is the origin.
    """
    return tol * (float(np.linalg.norm(b)) or 1.0)


def _cg_step(x: Vector, g: Vector, p: Vector, ap: Vector, p_ap: float, alpha: float) -> Iterate:
    """Conjugate-gradient step: minimise within the current face, leaving it unchanged.

    The caller has already checked ``alpha`` feasible, so no projection is needed
    and the gradient updates linearly. The next direction is the new free gradient
    made ``A``-conjugate to ``p`` by one Gram-Schmidt sweep. Costs no Hessian
    product of its own — ``ap`` is the one the caller computed to form ``alpha``.
    """
    x = x - alpha * p
    g = g - alpha * ap
    phi = _free_gradient(x, g)
    beta_gs = float(phi @ ap) / p_ap
    return x, g, phi - beta_gs * p


def _expansion_step(
    matvec: MatVec,
    b: Vector,
    x: Vector,
    g: Vector,
    p: Vector,
    ap: Vector,
    alpha_f: float,
    alpha_bar: float,
) -> Iterate:
    """Expansion step: walk to the nearest bound, then one projected-gradient move.

    Taken when the conjugate-gradient step would leave the feasible set. The
    iterate advances only as far as the bound (``alpha_f``) and then takes one
    fixed-step ``alpha_bar`` projected-gradient move, which *adds* the newly
    active constraints. The projection is non-linear, so the gradient is
    recomputed rather than updated — the extra Hessian product this step costs —
    and CG restarts from the new free gradient.
    """
    x_half = x - alpha_f * p
    g = g - alpha_f * ap
    phi_half = _free_gradient(x_half, g)
    x = np.maximum(x_half - alpha_bar * phi_half, 0.0)
    g = matvec(x) - b
    return x, g, _free_gradient(x, g)


def _proportioning_step(matvec: MatVec, x: Vector, g: Vector, beta: Vector) -> Iterate:
    """Proportioning step: release active constraints along the chopped gradient.

    Taken on a disproportional iterate. ``beta`` is non-positive on the active set
    and zero on the free set, and the exact line minimiser along it has
    ``alpha >= 0``, so the move points inward and stays feasible without a
    projection — the gradient therefore updates linearly. Costs one Hessian
    product; CG restarts afterwards.
    """
    ad = matvec(beta)
    alpha = float(g @ beta) / float(beta @ ad)
    x = x - alpha * beta
    g = g - alpha * ad
    return x, g, _free_gradient(x, g)


def _proportional_step(matvec: MatVec, b: Vector, iterate: Iterate, alpha_bar: float) -> tuple[Iterate, int, str]:
    """Take the conjugate-gradient or expansion move on a proportional iterate.

    Forms the trial conjugate-gradient step and compares it with the largest
    feasible step along ``p``: within the face it is a plain :func:`_cg_step`,
    otherwise the face must change and :func:`_expansion_step` walks to the bound
    and projects.

    Args:
        matvec: The action ``v -> A v`` of the SPD operator.
        b: The linear term ``b``.
        iterate: The current ``(x, g, p)``.
        alpha_bar: The fixed projected-gradient step of the expansion move.

    Returns:
        The next ``(x, g, p)``, the Hessian products consumed (one for the
        conjugate-gradient step, two for the expansion step), and which move was
        taken — ``"cg"`` or ``"expansion"`` — for the result's step counters.
    """
    x, g, p = iterate
    ap = matvec(p)
    p_ap = float(p @ ap)
    alpha_cg = float(g @ p) / p_ap
    alpha_f = _max_feasible_step(x, p)
    if alpha_cg <= alpha_f:
        return _cg_step(x, g, p, ap, p_ap, alpha_cg), 1, "cg"
    return _expansion_step(matvec, b, x, g, p, ap, alpha_f, alpha_bar), 2, "expansion"


def _resolve_alpha_bar(a: SymmetricOperator, alpha_bar: float | None, seed: int) -> float:
    """Return the fixed projected-gradient step, estimating ``1/||A||`` when unset.

    The convergence proof requires ``alpha_bar in (0, 2/||A||]``. Power iteration
    approaches ``||A|| = lambda_max`` from below, so ``2/lambda_est`` could exceed
    the bound; the conservative default ``1/lambda_est`` stays safely inside it for
    any estimate with ``lambda_est >= lambda_max / 2``.

    Args:
        a: The SPD operator ``A``.
        alpha_bar: An explicit step (returned as-is after validation), or ``None``
            for the ``1/lambda_max`` estimate.
        seed: Seed of the power-iteration test vector, for reproducibility.

    Returns:
        The positive fixed step ``alpha_bar``.

    Raises:
        ValueError: When an explicit ``alpha_bar`` is not strictly positive.
    """
    if alpha_bar is not None:
        if alpha_bar <= 0.0:
            msg = f"alpha_bar must be strictly positive; got {alpha_bar:.2e}"
            raise ValueError(msg)
        return float(alpha_bar)
    lam_max, _ = power_iteration(a, seed=seed)
    return 1.0 / float(lam_max)


@dataclass(frozen=True)
class MPRGPConfig:
    """Configuration of the MPRGP solver (:class:`MPRGP`).

    Attributes:
        tol: Relative stopping tolerance on the projected gradient — the loop
            exits when ``||phi(x) + beta(x)|| <= tol * ||b||`` (``||b||`` replaced
            by ``1`` when ``b = 0``), which certifies the KKT conditions.
        gamma: Proportioning constant ``Gamma > 0`` balancing expansion against
            proportioning. ``1.0`` is the standard, near-optimal choice; larger
            values expand more eagerly, smaller ones release more eagerly.
        alpha_bar: Fixed projected-gradient step, which must satisfy
            ``0 < alpha_bar <= 2/||A||`` for convergence. ``None`` estimates the
            safe default ``1/||A||`` by matrix-free power iteration.
        max_iter: Iteration cap; the current iterate is returned with
            ``converged=False`` when it is hit.
        seed: Seed of the power-iteration ``||A||`` estimate (only used when
            ``alpha_bar is None``), fixed so a solve is reproducible.

    Raises:
        ValueError: When ``gamma`` is not strictly positive.
    """

    tol: float = 1e-8
    gamma: float = 1.0
    alpha_bar: float | None = None
    max_iter: int = 100_000
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate that the proportioning constant is strictly positive."""
        if self.gamma <= 0.0:
            msg = f"gamma must be strictly positive; got {self.gamma:.2e}"
            raise ValueError(msg)


#: Shared default so ``MPRGPConfig()`` is not called in argument defaults (ruff B008).
_DEFAULT_MPRGP = MPRGPConfig()


@dataclass(frozen=True)
class MPRGPResult:
    """Outcome of an MPRGP solve.

    The iteration counts are broken out by move because they carry the algorithm's
    signature: expansion and proportioning steps are the ones that change the
    active set, while a run of conjugate-gradient steps is plain CG on a fixed
    face. ``hessian_products`` is the honest cost of a matrix-free method — one
    product per CG or proportioning step, two per expansion step, plus one for the
    initial gradient.

    Attributes:
        x: The minimiser (or the final iterate if ``converged`` is False).
        iterations: Total MPRGP steps taken (the sum of the three move counts).
        hessian_products: Number of operator matrix-vector products consumed.
        cg_steps: Conjugate-gradient (minimisation-within-the-face) steps.
        expansion_steps: Expansion (bound-hitting projected-gradient) steps.
        proportioning_steps: Proportioning (constraint-releasing) steps.
        converged: True when the projected-gradient stopping test was met; False
            when ``max_iter`` stopped the loop first.
        free: Boolean mask of the final free set (``x > 0``).
    """

    x: Vector
    iterations: int
    hessian_products: int
    cg_steps: int
    expansion_steps: int
    proportioning_steps: int
    converged: bool
    free: NDArray[np.bool_]


@dataclass(frozen=True)
class MPRGP:
    """The MPRGP solver for the non-negative quadratic program.

    A matrix-free, factorisation-free alternative to
    :class:`nncg.solver.ActiveSetSolver` on the bound-constrained problem
    ``min_{x>=0} 1/2 x^T A x - b^T x``: it interleaves conjugate-gradient,
    expansion and proportioning steps under the proportioning test, never forming
    or factorising ``A``. Holds only its :class:`MPRGPConfig`; the operator and
    right-hand side are passed to :meth:`solve`.

    Attributes:
        config: Solver configuration (tolerance, proportioning constant,
            projected-gradient step, iteration cap, seed).

    Examples:
        The same operator interface as the active-set loop:

        >>> import numpy as np
        >>> from cvx.linalg import DenseOperator
        >>> from nncg import MPRGP, MPRGPConfig, kkt_violation
        >>> a = DenseOperator(np.array([[2.0, 0.0], [0.0, 2.0]]))
        >>> b = np.array([2.0, -2.0])
        >>> res = MPRGP().solve(a, b)
        >>> res.converged
        True
        >>> bool(np.allclose(res.x, [1.0, 0.0]))
        True
        >>> round(kkt_violation(a, b, res.x), 12)
        0.0

        The counts break the run down by move, and always sum to
        ``iterations``; ``hessian_products`` is the honest matrix-free cost —
        at least one product per step, plus the initial gradient:

        >>> res.iterations == res.cg_steps + res.expansion_steps + res.proportioning_steps
        True
        >>> res.hessian_products > res.iterations
        True

        Nothing is factorised, so the whole configuration is a handful of
        scalars — here a tighter tolerance and an explicit projected-gradient
        step, which skips the power-iteration estimate of ``1/||A||``:

        >>> tight = MPRGP(config=MPRGPConfig(tol=1e-12, alpha_bar=0.5)).solve(a, b)
        >>> tight.converged
        True
        >>> bool(np.allclose(tight.x, [1.0, 0.0]))
        True
    """

    config: MPRGPConfig = _DEFAULT_MPRGP

    def solve(self, a: SymmetricOperator, b: Vector, x0: Vector | None = None) -> MPRGPResult:
        """Minimise ``1/2 x^T A x - b^T x`` over ``x >= 0`` by MPRGP.

        Args:
            a: The SPD operator ``A`` (a :class:`cvx.linalg.SymmetricOperator`) —
                ``DenseOperator`` for an explicit array, ``GramOperator(M, ridge)``
                for ``A = M^T M + ridge I`` whose Gram matrix is never formed.
            b: The linear term ``b``.
            x0: Optional feasible warm start; it is projected onto ``x >= 0`` and
                the iteration begins there. ``None`` starts from the origin.

        Returns:
            An :class:`MPRGPResult`; ``converged`` is True iff the projected
            gradient fell below ``config.tol * ||b||``, which certifies the unique
            global minimiser.

        Raises:
            TypeError: When ``a`` is not a :class:`cvx.linalg.SymmetricOperator`.
            ValueError: When the operator dimension does not match ``len(b)``, or
                when ``config.alpha_bar`` is set but not strictly positive.
        """
        _require_operator(a, b)
        cfg = self.config
        alpha_bar = _resolve_alpha_bar(a, cfg.alpha_bar, cfg.seed)
        return _mprgp(a.matvec, b, x0, alpha_bar, cfg.gamma, cfg.tol, cfg.max_iter)


def _mprgp(
    matvec: MatVec,
    b: Vector,
    x0: Vector | None,
    alpha_bar: float,
    gamma: float,
    tol: float,
    max_iter: int,
) -> MPRGPResult:
    """Run the MPRGP iteration and assemble its :class:`MPRGPResult`.

    The pure algorithm behind :meth:`MPRGP.solve`: it takes the resolved operator
    action and step bound and owns the loop — the projected-gradient stopping
    test and the proportioning switch — while the three moves themselves live in
    :func:`_cg_step`, :func:`_expansion_step` and :func:`_proportioning_step`,
    with the first two selected by :func:`_proportional_step`. Kept as a plain
    function (operator access reduced to ``matvec``) so the numerics can be
    exercised directly.

    Args:
        matvec: The action ``v -> A v`` of the SPD operator.
        b: The linear term ``b``.
        x0: Optional warm start (projected onto ``x >= 0``); ``None`` starts at 0.
        alpha_bar: The fixed projected-gradient step, ``0 < alpha_bar <= 2/||A||``.
        gamma: The proportioning constant ``Gamma > 0``.
        tol: Relative projected-gradient stopping tolerance (against ``||b||``).
        max_iter: Iteration cap.

    Returns:
        The completed :class:`MPRGPResult`.
    """
    x, g, p = _initial_iterate(matvec, b, x0)
    products = 1  # the gradient A x - b of the initial iterate
    stop = _stopping_threshold(b, tol)
    counts = {"cg": 0, "expansion": 0, "proportioning": 0}
    iterations = 0
    converged = False

    while iterations < max_iter:
        phi = _free_gradient(x, g)
        beta = _chopped_gradient(x, g)
        if float(np.linalg.norm(phi + beta)) <= stop:
            converged = True
            break
        iterations += 1

        phi_tilde = _reduced_free_gradient(x, g, alpha_bar)
        if float(beta @ beta) <= gamma * gamma * float(phi_tilde @ phi):
            # Proportional iterate: minimise within the face, or expand it.
            (x, g, p), used, move = _proportional_step(matvec, b, (x, g, p), alpha_bar)
            products += used
            counts[move] += 1
        else:
            # Disproportional iterate: release constraints along the chopped gradient.
            x, g, p = _proportioning_step(matvec, x, g, beta)
            products += 1
            counts["proportioning"] += 1

    return MPRGPResult(
        x=x,
        iterations=iterations,
        hessian_products=products,
        cg_steps=counts["cg"],
        expansion_steps=counts["expansion"],
        proportioning_steps=counts["proportioning"],
        converged=converged,
        free=x > 0.0,
    )
