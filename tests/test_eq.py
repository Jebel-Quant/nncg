"""Tests of the equality-augmented solver (B x = c via Schur complement)."""

import numpy as np
import pytest
from cvx.linalg import DenseOperator

from nncg import solve_nnqp_eq
from tests.problems import make_eq_problem, make_scaled_eq_problem


@pytest.mark.parametrize("p", [1, 3, 8])
@pytest.mark.parametrize("kappa", [1e2, 1e4])
def test_recovers_planted_eq_optimum(p: int, kappa: float) -> None:
    """The Schur-complement loop recovers the planted optimum for general B."""
    a, b, b_eq, c_eq, x_star, _, _ = make_eq_problem(80, kappa, p, seed=p)
    res = solve_nnqp_eq(DenseOperator(a), b, b_eq, c_eq)
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
    from tests.problems import make_problem

    a, b, x_star, _ = make_problem(60, 1e3, seed=0)
    ones = np.ones((1, 60))
    beta = np.array([float(x_star.sum())])
    res = solve_nnqp_eq(DenseOperator(a), b, ones, beta)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert res.lam is not None
    assert abs(float(res.lam[0])) < 1e-6
    assert abs(float(res.x.sum()) - float(beta[0])) < 1e-9


@pytest.mark.parametrize("inner", ["cg", "pcg"])
def test_eq_warm_start_support_stable_single_step(inner: str) -> None:
    """Across a support-stable step the warm eq loop takes one outer step and cuts inner iters.

    Both iterative inners consume the ``v0`` warm seed: the warm solve makes
    one outer step (the warm free set is already optimal) and its inner count
    falls below the cold solve's — the ``v0`` block converges from a near
    solution while the ``v1`` columns stay cold on both.
    """
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(80, 1e3, 3, seed=2)
    op = DenseOperator(a)
    first = solve_nnqp_eq(op, b, b_eq, c_eq, inner=inner)
    delta = 1e-4 * np.linalg.norm(b) * np.ones_like(b) / np.sqrt(len(b))
    second_cold = solve_nnqp_eq(op, b + delta, b_eq, c_eq, inner=inner)
    assert np.array_equal(second_cold.free, first.free)  # support-stable step
    second_warm = solve_nnqp_eq(op, b + delta, b_eq, c_eq, inner=inner, warm=(first.free, first.x))
    assert second_warm.converged
    assert second_warm.outer == 1
    assert second_warm.inner < second_cold.inner
    assert np.max(np.abs(second_warm.x - second_cold.x)) < 1e-6
    assert np.linalg.norm(b_eq @ second_warm.x - c_eq) < 1e-9


def test_eq_warm_start_exact_single_outer_step() -> None:
    """``inner="exact"`` has no inner seed, but the warm free set still gives one outer step.

    The direct solve has nothing to warm-start, so the reduction is purely the
    outer loop's: starting from the previous (already optimal) free set, the
    warm solve converges in a single outer step where the cold solve needs
    several — this pins the intentional "no inner seed for exact" behaviour.
    """
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(80, 1e3, 3, seed=2)
    op = DenseOperator(a)
    first = solve_nnqp_eq(op, b, b_eq, c_eq, inner="exact")
    delta = 1e-4 * np.linalg.norm(b) * np.ones_like(b) / np.sqrt(len(b))
    cold = solve_nnqp_eq(op, b + delta, b_eq, c_eq, inner="exact")
    assert np.array_equal(cold.free, first.free)  # support-stable step
    assert cold.outer > 1  # cold starts from the full free set and must shrink
    warm = solve_nnqp_eq(op, b + delta, b_eq, c_eq, inner="exact", warm=(first.free, first.x))
    assert warm.converged
    assert warm.outer == 1
    assert np.max(np.abs(warm.x - cold.x)) < 1e-6
    assert np.linalg.norm(b_eq @ warm.x - c_eq) < 1e-9


def test_eq_warm_start_survives_support_drift() -> None:
    """A warm start from a drifted support still reaches the right eq optimum."""
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(80, 1e3, 3, seed=3)
    op = DenseOperator(a)
    first = solve_nnqp_eq(op, b, b_eq, c_eq)
    b2 = b + 0.3 * np.linalg.norm(b) * np.random.default_rng(0).standard_normal(len(b)) / np.sqrt(len(b))
    cold = solve_nnqp_eq(op, b2, b_eq, c_eq)
    warm = solve_nnqp_eq(op, b2, b_eq, c_eq, warm=(first.free, first.x))
    assert warm.converged
    assert np.max(np.abs(warm.x - cold.x)) < 1e-6
    assert np.linalg.norm(b_eq @ warm.x - c_eq) < 1e-9


def test_eq_exact_inner_matches_cg() -> None:
    """The direct (``inner="exact"``) eq solve matches the CG solve.

    Both recover the planted optimum and settle on the same free set — the
    equality analogue of the inexactness lemma: the inner accuracy does not
    change the sign decisions of the outer loop.
    """
    a, b, b_eq, c_eq, x_star, _, _ = make_eq_problem(80, 1e4, 3, seed=5)
    op = DenseOperator(a)
    r_cg = solve_nnqp_eq(op, b, b_eq, c_eq)
    r_ex = solve_nnqp_eq(op, b, b_eq, c_eq, inner="exact")
    assert r_ex.converged
    assert np.array_equal(r_ex.free, r_cg.free)
    assert np.max(np.abs(r_ex.x - x_star)) < 1e-6
    assert np.linalg.norm(b_eq @ r_ex.x - c_eq) < 1e-9


def test_eq_pcg_inner_recovers_optimum() -> None:
    """The Jacobi-preconditioned eq solve recovers the same planted optimum."""
    a, b, b_eq, c_eq, x_star, _, _ = make_eq_problem(80, 1e4, 3, seed=6)
    r_pcg = solve_nnqp_eq(DenseOperator(a), b, b_eq, c_eq, inner="pcg")
    assert r_pcg.converged
    assert np.max(np.abs(r_pcg.x - x_star)) < 1e-6
    assert np.linalg.norm(b_eq @ r_pcg.x - c_eq) < 1e-9


def test_eq_nystrom_inner_recovers_optimum() -> None:
    """The Nyström-preconditioned eq solve recovers the same planted optimum.

    Exercises the per-free-set sketch cache: each saddle step drives ``p + 1``
    right-hand sides through the same ``A_F``, and the sketch is built once.
    """
    a, b, b_eq, c_eq, x_star, _, _ = make_eq_problem(80, 1e4, 3, seed=6)
    r_ny = solve_nnqp_eq(DenseOperator(a), b, b_eq, c_eq, inner="nystrom", nystrom_rank=15)
    assert r_ny.converged
    assert np.max(np.abs(r_ny.x - x_star)) < 1e-6
    assert np.linalg.norm(b_eq @ r_ny.x - c_eq) < 1e-9


def test_eq_pcg_beats_cg_under_diagonal_scaling() -> None:
    """On a diagonally ill-scaled eq problem PCG needs fewer inner iterations."""
    a, b, b_eq, c_eq, x_star, _, _ = make_scaled_eq_problem(80, 1e2, 1e6, 3, seed=7)
    op = DenseOperator(a)
    r_cg = solve_nnqp_eq(op, b, b_eq, c_eq, inner="cg")
    r_pcg = solve_nnqp_eq(op, b, b_eq, c_eq, inner="pcg")
    assert r_pcg.converged
    assert np.max(np.abs(r_pcg.x - x_star)) < 1e-6
    assert np.max(np.abs(r_cg.x - x_star)) < 1e-6
    # A comfortable margin, not a bare `<`: the 1e6 diagonal spread makes the
    # Jacobi win large enough to survive BLAS-dependent iteration-count drift.
    assert r_pcg.inner <= 0.7 * r_cg.inner


def test_eq_track_records_trajectory() -> None:
    """``track=True`` records the visited free sets, ending on the converged one."""
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(60, 1e3, 3, seed=8)
    res = solve_nnqp_eq(DenseOperator(a), b, b_eq, c_eq, track=True)
    assert res.converged
    assert res.traj is not None
    assert len(res.traj) == res.outer  # one recorded free set per outer step
    assert np.array_equal(res.traj[-1], np.flatnonzero(res.free))  # last = converged support


def test_eq_max_outer_caps_the_loop() -> None:
    """``max_outer`` stops the loop early and reports non-convergence."""
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(80, 1e5, 3, seed=9)
    res = solve_nnqp_eq(DenseOperator(a), b, b_eq, c_eq, max_outer=1)
    assert not res.converged
    assert res.outer == 1


def test_eq_rejects_unknown_inner() -> None:
    """An unrecognised inner solver is rejected up front, not run as CG."""
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(20, 1e2, 1, seed=0)
    with pytest.raises(ValueError, match="inner must be"):
        solve_nnqp_eq(DenseOperator(a), b, b_eq, c_eq, inner="nope")  # type: ignore[arg-type]


def test_eq_reduced_gradient_certifies() -> None:
    """At the exit, x >= 0 and s = Ax - b - B^T lam >= 0 hold to tolerance.

    The certificate tolerance is 1e-6, not the solver's tol=1e-8: the dual
    test only guards bound indices, while on free indices s carries the inner
    CG residual, which varies with the NumPy/BLAS version.
    """
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(60, 1e3, 3, seed=1)
    res = solve_nnqp_eq(DenseOperator(a), b, b_eq, c_eq)
    assert res.lam is not None
    s = a @ res.x - b - b_eq.T @ res.lam
    assert float(np.min(res.x)) > -1e-6
    assert float(np.min(s)) > -1e-6
    assert float(np.max(np.abs(res.x * s))) < 1e-6
