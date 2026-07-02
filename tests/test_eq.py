"""Tests of the equality-augmented solver (B x = c via Schur complement)."""

import numpy as np
import pytest

from nncg import make_eq_problem, solve_nnqp_eq


@pytest.mark.parametrize("p", [1, 3, 8])
@pytest.mark.parametrize("kappa", [1e2, 1e4])
def test_recovers_planted_eq_optimum(p: int, kappa: float) -> None:
    """The Schur-complement loop recovers the planted optimum for general B."""
    a, b, b_eq, c_eq, x_star, _, _ = make_eq_problem(80, kappa, p, seed=p)
    res = solve_nnqp_eq(a, b, b_eq, c_eq)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert np.linalg.norm(b_eq @ res.x - c_eq) < 1e-9  # feasible to machine precision
    assert res.lam is not None
    assert res.lam.shape == (p,)


def test_p_one_is_the_single_normalisation() -> None:
    """P = 1 with B = 1^T reproduces the bound-constrained solve on its own budget.

    With lambda* = 0 the bound-constrained optimum x* satisfies the KKT system
    of the equality-augmented problem whose budget is beta = 1^T x*, so the
    Schur-complement loop must return x* with a vanishing multiplier.
    """
    from nncg import make_problem

    a, b, x_star, _ = make_problem(60, 1e3, seed=0)
    ones = np.ones((1, 60))
    beta = np.array([float(x_star.sum())])
    res = solve_nnqp_eq(a, b, ones, beta)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert res.lam is not None
    assert abs(float(res.lam[0])) < 1e-6
    assert abs(float(res.x.sum()) - float(beta[0])) < 1e-9


def test_eq_reduced_gradient_certifies() -> None:
    """At the exit, x >= 0 and s = Ax - b - B^T lam >= 0 hold to tolerance.

    On bound indices the loop guarantees s >= -tol; on FREE indices s vanishes
    only to the inner solver's accuracy, which scales with ||b|| (cg_tol is a
    relative residual tolerance) — so the assertions must be scale-aware.
    """
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(60, 1e3, 3, seed=1)
    res = solve_nnqp_eq(a, b, b_eq, c_eq)
    assert res.lam is not None
    s = a @ res.x - b - b_eq.T @ res.lam
    scale = 1.0 + float(np.linalg.norm(b))
    assert float(np.min(res.x)) > -1e-8
    assert float(np.min(s)) > -1e-9 * scale
    assert float(np.max(np.abs(res.x * s))) < 1e-9 * scale
