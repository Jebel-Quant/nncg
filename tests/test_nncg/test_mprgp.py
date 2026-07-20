"""Tests of the MPRGP solver: the projection-based companion to the active-set loop.

Covers the bound-constrained solve on the shared planted-optimum generators,
the matrix-free operator interface (``A`` enters only as a
``cvx.linalg.SymmetricOperator``), agreement with
:class:`nncg.solver.ActiveSetSolver` on the same problem, the three-move
bookkeeping (conjugate-gradient / expansion / proportioning) that gives the
Hessian-product cost, warm starting, the ``max_iter`` non-convergence contract,
config validation, and unconditional termination on the anti-correlated
adversarial family.

MPRGP is a first-order method: its iterate accuracy scales with the conditioning
of ``A`` (it never solves a free block exactly, unlike the active-set loop), so
the recovery tests either stay at moderate conditioning or tighten ``tol`` for
the ill-conditioned case, and the certificate is asserted relative to ``||b||``.
"""

import numpy as np
import pytest
from cvx.linalg import DenseOperator, GramOperator

from nncg import (
    MPRGP,
    MPRGPConfig,
    MPRGPResult,
    kkt_violation,
    solve_nnqp,
    solve_nnqp_mprgp,
)
from nncg.mprgp import (
    _chopped_gradient,
    _free_gradient,
    _max_feasible_step,
    _reduced_free_gradient,
)
from tests.problems import make_adversarial, make_problem


def _projected_gradient_norm(op: DenseOperator, b: np.ndarray, x: np.ndarray) -> float:
    """Norm of the projected gradient ``phi(x) + beta(x)`` — MPRGP's stopping quantity."""
    g = op.matvec(x) - b
    return float(np.linalg.norm(_free_gradient(x, g) + _chopped_gradient(x, g)))


# --------------------------------------------------------------------------
# Recovery of the planted optimum
# --------------------------------------------------------------------------


def test_mprgp_class_api_and_result_fields() -> None:
    """``MPRGP`` bundles the config and returns a fully populated ``MPRGPResult``."""
    a, b, x_star, _ = make_problem(60, 1e3, seed=0)
    res = MPRGP(config=MPRGPConfig()).solve(DenseOperator(a), b)
    assert isinstance(res, MPRGPResult)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    # the three move counts partition the iteration total
    assert res.iterations == res.cg_steps + res.expansion_steps + res.proportioning_steps
    # one product per CG/proportioning step, two per expansion, plus the initial gradient
    assert res.hessian_products == 1 + res.cg_steps + 2 * res.expansion_steps + res.proportioning_steps


@pytest.mark.parametrize("kappa", [1e1, 1e3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_planted_optimum(kappa: float, seed: int) -> None:
    """MPRGP recovers the planted optimum across moderate condition numbers and seeds."""
    a, b, x_star, _ = make_problem(80, kappa, seed=seed)
    op = DenseOperator(a)
    res = MPRGP().solve(op, b)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    # the certificate scales with ||b||, so it is asserted relative to it
    assert kkt_violation(op, b, res.x) < 1e-7 * (1.0 + float(np.linalg.norm(b)))


def test_ill_conditioned_needs_tighter_tol() -> None:
    """At kappa = 1e5 the first-order iterate needs a tighter tol to reach 1e-6 in x.

    The projected gradient falls below ``tol * ||b||`` regardless, but a first-order
    method's iterate error carries a factor of the condition number, so the default
    ``tol=1e-8`` leaves x short of ``1e-6`` while ``tol=1e-11`` recovers it.
    """
    a, b, x_star, _ = make_problem(80, 1e5, seed=0)
    op = DenseOperator(a)
    loose = MPRGP(MPRGPConfig(tol=1e-8)).solve(op, b)
    assert loose.converged
    assert _projected_gradient_norm(op, b, loose.x) <= 1e-8 * float(np.linalg.norm(b)) * (1 + 1e-6)
    tight = MPRGP(MPRGPConfig(tol=1e-11)).solve(op, b)
    assert tight.converged
    assert np.max(np.abs(tight.x - x_star)) < 1e-6


def test_free_mask_matches_support() -> None:
    """The final free mask is the support of the planted optimum."""
    a, b, x_star, _ = make_problem(60, 1e2, seed=3)
    res = MPRGP(MPRGPConfig(tol=1e-10)).solve(DenseOperator(a), b)
    assert np.array_equal(res.free, x_star > 0)


def test_matches_active_set_solver() -> None:
    """MPRGP and the active-set loop reach the same minimiser on the same problem."""
    a, b, _, _ = make_problem(80, 1e4, seed=1)
    active = solve_nnqp(a, b)
    mprgp = solve_nnqp_mprgp(a, b, tol=1e-11)
    assert mprgp.converged
    assert np.max(np.abs(mprgp.x - active.x)) < 1e-6


def test_zero_rhs_returns_origin() -> None:
    """With ``b = 0`` the origin is already optimal: zero iterations, empty free set."""
    a, _, _, _ = make_problem(40, 1e2, seed=0)
    res = MPRGP().solve(DenseOperator(a), np.zeros(40))
    assert res.converged
    assert res.iterations == 0
    assert np.array_equal(res.x, np.zeros(40))
    assert not res.free.any()


# --------------------------------------------------------------------------
# Matrix-free operator interface
# --------------------------------------------------------------------------


def test_gram_operator_recovers_planted_optimum() -> None:
    """MPRGP runs matrix-free on a Gram operator: ``A = M^T M + ridge I`` never formed."""
    rng = np.random.default_rng(0)
    n, m_rows, ridge = 60, 30, 1.0
    m = rng.standard_normal((m_rows, n))
    perm = rng.permutation(n)
    x_star = np.zeros(n)
    x_star[perm[: n // 2]] = rng.uniform(0.5, 1.5, size=n // 2)
    s_star = np.zeros(n)
    s_star[perm[n // 2 :]] = rng.uniform(0.5, 1.5, size=n - n // 2)
    b = m.T @ (m @ x_star) + ridge * x_star - s_star
    op = GramOperator(m, ridge=ridge)
    res = MPRGP(MPRGPConfig(tol=1e-10)).solve(op, b)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert kkt_violation(op, b, res.x) < 1e-6 * (1.0 + float(np.linalg.norm(b)))


def test_convenience_wraps_dense_array() -> None:
    """``solve_nnqp_mprgp`` wraps a plain SPD array in ``DenseOperator`` like the class."""
    a, b, x_star, _ = make_problem(60, 1e2, seed=2)
    res = solve_nnqp_mprgp(a, b, tol=1e-10)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6


# --------------------------------------------------------------------------
# Warm start
# --------------------------------------------------------------------------


def test_warm_start_reduces_iterations() -> None:
    """A feasible warm start near the optimum cuts the iteration count."""
    a, b, x_star, _ = make_problem(80, 1e3, seed=0)
    op = DenseOperator(a)
    cold = MPRGP().solve(op, b)
    warm = MPRGP().solve(op, b, x0=x_star)
    assert warm.converged
    assert warm.iterations < cold.iterations
    assert np.max(np.abs(warm.x - cold.x)) < 1e-6


def test_warm_start_is_projected_onto_feasible_set() -> None:
    """An infeasible ``x0`` is projected onto ``x >= 0`` rather than rejected."""
    a, b, x_star, _ = make_problem(60, 1e2, seed=1)
    x0 = x_star - 5.0  # drives many entries negative
    res = MPRGP(MPRGPConfig(tol=1e-10)).solve(DenseOperator(a), b, x0=x0)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6


# --------------------------------------------------------------------------
# max_iter contract and step bookkeeping
# --------------------------------------------------------------------------


def test_max_iter_cap_reports_nonconvergence() -> None:
    """An iteration cap short of convergence stops at exactly the cap, converged=False."""
    a, b, _, _ = make_problem(80, 1e4, seed=0)
    op = DenseOperator(a)
    full = MPRGP().solve(op, b).iterations
    assert full > 1
    cap = full // 2
    res = MPRGP(MPRGPConfig(max_iter=cap)).solve(op, b)
    assert not res.converged
    assert res.iterations == cap


def test_all_three_moves_are_exercised() -> None:
    """A non-trivial solve takes conjugate-gradient, expansion and proportioning steps.

    Expansion steps add constraints and proportioning steps release them; a
    bound-active optimum reached from the origin needs both, interleaved with the
    CG steps that minimise within each face — this pins that all three code paths
    run, not just the CG fast path.
    """
    a, b, _, _ = make_problem(80, 1e3, seed=0)
    res = MPRGP().solve(DenseOperator(a), b)
    assert res.cg_steps > 0
    assert res.expansion_steps > 0
    assert res.proportioning_steps > 0


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_explicit_alpha_bar_is_used() -> None:
    """An explicit ``alpha_bar`` skips the power-iteration estimate and still converges."""
    a, b, x_star, _ = make_problem(60, 1e2, seed=0)
    lam_max = float(np.linalg.eigvalsh(a)[-1])
    res = MPRGP(MPRGPConfig(alpha_bar=1.0 / lam_max, tol=1e-10)).solve(DenseOperator(a), b)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6


def test_nonpositive_alpha_bar_is_rejected() -> None:
    """A non-positive ``alpha_bar`` is refused (convergence needs ``alpha_bar > 0``)."""
    a, b, _, _ = make_problem(20, 1e2, seed=0)
    with pytest.raises(ValueError, match="alpha_bar"):
        MPRGP(MPRGPConfig(alpha_bar=0.0)).solve(DenseOperator(a), b)


def test_nonpositive_gamma_is_rejected() -> None:
    """A non-positive proportioning constant is refused at config construction."""
    with pytest.raises(ValueError, match="gamma"):
        MPRGPConfig(gamma=0.0)


def test_dimension_mismatch_is_rejected() -> None:
    """An operator whose dimension disagrees with len(b) is refused at entry."""
    a, b, _, _ = make_problem(20, 1e2, seed=0)
    with pytest.raises(ValueError, match="dimension"):
        MPRGP().solve(DenseOperator(a), b[:-1])


def test_dense_array_is_rejected_by_class() -> None:
    """The class refuses a bare array with a pointer to DenseOperator."""
    a, b, _, _ = make_problem(20, 1e2, seed=0)
    with pytest.raises(TypeError, match="DenseOperator"):
        MPRGP().solve(a, b)


# --------------------------------------------------------------------------
# Unconditional termination on the adversarial family
# --------------------------------------------------------------------------


def test_terminates_on_adversarial_family() -> None:
    """MPRGP converges and certifies on every anti-correlated adversarial seed.

    The anti-correlated design forces the active-set loop's Bland fallback; MPRGP
    is a monotone projection method with no working-set cycling to guard against,
    so it simply converges on all of them.
    """
    for seed in range(30):
        a, b = make_adversarial(20, seed=seed)
        op = DenseOperator(a)
        res = MPRGP(MPRGPConfig(max_iter=50_000)).solve(op, b)
        assert res.converged
        assert kkt_violation(op, b, res.x) < 1e-6 * (1.0 + float(np.linalg.norm(b)))


# --------------------------------------------------------------------------
# Gradient primitives
# --------------------------------------------------------------------------


def test_gradient_primitives_partition_the_gradient() -> None:
    """On the free set the gradient is the free gradient; on the active set, the chopped one.

    ``phi`` carries the gradient where ``x > 0`` (zero elsewhere) and ``beta`` the
    negative part where ``x = 0`` (zero elsewhere); the reduced free gradient caps
    ``phi`` by the feasible step ``x_i / alpha_bar`` on the free set.
    """
    x = np.array([0.0, 2.0, 0.0, 1.0])
    g = np.array([-3.0, 5.0, 4.0, -1.0])
    phi = _free_gradient(x, g)
    beta = _chopped_gradient(x, g)
    assert np.array_equal(phi, [0.0, 5.0, 0.0, -1.0])  # free set only
    assert np.array_equal(beta, [-3.0, 0.0, 0.0, 0.0])  # active set, negative part only
    # reduced free gradient caps a downhill free component by the distance to the bound
    alpha_bar = 4.0
    phi_tilde = _reduced_free_gradient(x, g, alpha_bar)
    assert np.array_equal(phi_tilde, [0.0, min(5.0, 2.0 / 4.0), 0.0, -1.0])


def test_max_feasible_step() -> None:
    """The feasible step is ``min_{p_i > 0} x_i / p_i``, unbounded when no component decreases."""
    x = np.array([2.0, 1.0, 3.0])
    assert _max_feasible_step(x, np.array([1.0, 4.0, 0.0])) == pytest.approx(min(2.0 / 1.0, 1.0 / 4.0))
    assert _max_feasible_step(x, np.array([-1.0, 0.0, -2.0])) == np.inf
