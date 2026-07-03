"""Tests of the planted-optimum problem generators."""

import numpy as np
from cvx.linalg import DenseOperator
from problems import make_adversarial, make_eq_problem, make_problem, make_scaled_problem

from nncg import kkt_violation


def test_make_problem_plants_kkt_point() -> None:
    """The planted pair satisfies the KKT system exactly."""
    a, b, x_star, s_star = make_problem(50, 1e3, seed=0)
    assert np.allclose(a @ x_star - b, s_star)
    assert float(np.min(x_star)) >= 0.0
    assert float(np.min(s_star)) >= 0.0
    assert float(np.max(np.abs(x_star * s_star))) == 0.0
    assert kkt_violation(DenseOperator(a), b, x_star) < 1e-12


def test_make_problem_condition_number() -> None:
    """The spectrum spans [1, kappa] as prescribed."""
    kappa = 1e4
    a, _, _, _ = make_problem(50, kappa, seed=1)
    eig = np.linalg.eigvalsh(a)
    assert eig[0] > 0
    assert np.isclose(eig[-1] / eig[0], kappa, rtol=1e-6)


def test_make_eq_problem_plants_kkt_point() -> None:
    """The planted equality-augmented triple satisfies its KKT system."""
    a, b, b_eq, c_eq, x_star, lam_star, s_star = make_eq_problem(50, 1e3, 3, seed=0)
    assert np.allclose(a @ x_star - b_eq.T @ lam_star - b, s_star)
    assert np.allclose(b_eq @ x_star, c_eq)
    assert float(np.max(np.abs(x_star * s_star))) == 0.0


def test_make_adversarial_is_spd() -> None:
    """The adversarial design is strictly positive definite (a P-matrix)."""
    a, b = make_adversarial(20, seed=0)
    assert np.linalg.eigvalsh(a)[0] > 0
    assert b.shape == (20,)


def test_make_scaled_problem_spread() -> None:
    """The scaled operator's conditioning grows with the diagonal spread."""
    a1, _, _ = make_scaled_problem(40, 10.0, 1.0, seed=0)
    a2, _, _ = make_scaled_problem(40, 10.0, 1e4, seed=0)
    k1 = float(np.linalg.cond(a1))
    k2 = float(np.linalg.cond(a2))
    assert k2 > 10 * k1
