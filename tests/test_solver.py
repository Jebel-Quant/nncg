"""Tests of the active-set loop: the paper's numerical study as a test suite."""

import numpy as np
import pytest
from cvx.linalg import DenseOperator

from nncg import kkt_violation, solve_nnqp
from tests.problems import make_problem, make_scaled_problem


@pytest.mark.parametrize("kappa", [1e1, 1e3, 1e5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_planted_optimum(kappa: float, seed: int) -> None:
    """The loop recovers the planted optimum across condition numbers and seeds."""
    a, b, x_star, _ = make_problem(80, kappa, seed=seed)
    res = solve_nnqp(DenseOperator(a), b)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    # the complementarity products scale with ||b||, so the certificate is
    # asserted relative to the problem's magnitude
    assert kkt_violation(DenseOperator(a), b, res.x) < 1e-7 * (1.0 + float(np.linalg.norm(b)))


def test_outer_count_small_and_fallback_dormant() -> None:
    """On generic data the outer count is single-digit and the fallback dormant."""
    for seed in range(3):
        a, b, _, _ = make_problem(80, 1e4, seed=seed)
        res = solve_nnqp(DenseOperator(a), b)
        assert res.outer < 10
        assert res.fallback == 0


def test_free_mask_matches_support() -> None:
    """The final free mask is the support of the planted optimum."""
    a, b, x_star, _ = make_problem(60, 1e2, seed=3)
    res = solve_nnqp(DenseOperator(a), b)
    assert np.array_equal(res.free, x_star > 0)


def test_exact_inner_matches_cg_trajectory() -> None:
    """CG and exact inner solves visit the same free sets (inexactness lemma)."""
    for seed in range(3):
        a, b, _, _ = make_problem(60, 1e4, seed=seed)
        r_cg = solve_nnqp(DenseOperator(a), b, track=True)
        r_ex = solve_nnqp(DenseOperator(a), b, inner="exact", track=True)
        assert r_cg.traj == r_ex.traj


def test_pcg_inner_solve() -> None:
    """The Jacobi-PCG inner solver reaches the same optimum, cheaper when scaled."""
    a, b, x_star = make_scaled_problem(80, 50.0, 1e4, seed=1)
    r_cg = solve_nnqp(DenseOperator(a), b)
    r_pcg = solve_nnqp(DenseOperator(a), b, inner="pcg")
    assert np.max(np.abs(r_cg.x - x_star)) < 1e-6
    assert np.max(np.abs(r_pcg.x - x_star)) < 1e-6
    assert r_pcg.inner < r_cg.inner


def test_max_outer_cap_reports_nonconvergence() -> None:
    """An outer-step cap stops the loop with converged=False."""
    a, b, _, _ = make_problem(60, 1e3, seed=0)
    res = solve_nnqp(DenseOperator(a), b, max_outer=1)
    assert not res.converged
    assert res.outer == 1


def test_warm_start_support_stable_single_step() -> None:
    """Across a support-stable parameter step the warm loop takes one outer step."""
    a, b, _, _ = make_problem(80, 1e3, seed=0)
    first = solve_nnqp(DenseOperator(a), b)
    delta = 1e-4 * np.linalg.norm(b) * np.ones_like(b) / np.sqrt(len(b))
    second_cold = solve_nnqp(DenseOperator(a), b + delta)
    assert np.array_equal(second_cold.free, first.free)  # support-stable step
    second_warm = solve_nnqp(DenseOperator(a), b + delta, warm=(first.free, first.x))
    assert second_warm.converged
    assert second_warm.outer == 1
    assert second_warm.inner < second_cold.inner
    assert np.max(np.abs(second_warm.x - second_cold.x)) < 1e-6


def test_warm_start_survives_support_drift() -> None:
    """A warm start from a drifted support still reaches the right optimum."""
    a, b, _, _ = make_problem(80, 1e3, seed=1)
    first = solve_nnqp(DenseOperator(a), b)
    b2 = b + 0.3 * np.linalg.norm(b) * np.random.default_rng(0).standard_normal(len(b)) / np.sqrt(len(b))
    cold = solve_nnqp(DenseOperator(a), b2)
    warm = solve_nnqp(DenseOperator(a), b2, warm=(first.free, first.x))
    assert warm.converged
    assert np.max(np.abs(warm.x - cold.x)) < 1e-6


def test_rank_deficient_regularised_gram() -> None:
    """With m < n, any alpha > 0 restores well-posedness and recovery."""
    n, m, k = 80, 40, 20
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((m, n)) / np.sqrt(m)
    a0 = mat.T @ mat  # PSD, rank m < n
    perm = rng.permutation(n)
    x_star = np.zeros(n)
    x_star[perm[:k]] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
    for alpha in (0.05, 0.2):
        a = (1 - alpha) * a0 + alpha * np.eye(n)
        b = a @ x_star - s_star
        res = solve_nnqp(DenseOperator(a), b)
        assert res.converged
        assert np.max(np.abs(res.x - x_star)) < 1e-6
