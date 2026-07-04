"""Cross-checks that the alternative solvers agree with ``nncg`` on the plant.

Every baseline in :mod:`tests.baselines` solves the same planted problems as
:mod:`nncg.solver`, so the honest test is that each recovers the *same* known
optimum. OSQP and Clarabel are imported lazily and skipped when absent; the
pure-NumPy Lawson-Hanson and Duchi routines always run.
"""

import numpy as np
import pytest
from cvx.linalg import DenseOperator

from nncg import kkt_violation, solve_nnqp, solve_nnqp_eq
from tests import baselines as bl
from tests.problems import make_problem, make_simplex_problem

_ERR = 1e-5  # agreement tolerance on ||x - x_star|| across solvers


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_osqp_recovers_bound_plant(seed: int) -> None:
    """OSQP recovers the bound-only planted optimum and certifies it."""
    osqp = pytest.importorskip("osqp")  # noqa: F841
    a, b, x_star, _ = make_problem(60, 1e3, seed=seed)
    op = DenseOperator(a)
    r = bl.solve_osqp(op, b)
    assert np.max(np.abs(r.x - x_star)) < _ERR
    assert kkt_violation(op, b, r.x) < 1e-5 * (1.0 + float(np.linalg.norm(b)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clarabel_recovers_bound_plant(seed: int) -> None:
    """Clarabel recovers the bound-only planted optimum and certifies it."""
    clarabel = pytest.importorskip("clarabel")  # noqa: F841
    a, b, x_star, _ = make_problem(60, 1e3, seed=seed)
    op = DenseOperator(a)
    r = bl.solve_clarabel(op, b)
    assert np.max(np.abs(r.x - x_star)) < _ERR
    assert kkt_violation(op, b, r.x) < 1e-5 * (1.0 + float(np.linalg.norm(b)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lawson_hanson_recovers_bound_plant(seed: int) -> None:
    """Lawson-Hanson (CG inner) recovers the plant and its support exactly."""
    a, b, x_star, _ = make_problem(60, 1e3, seed=seed)
    op = DenseOperator(a)
    r = bl.solve_lawson_hanson(op, b)
    assert r.status == "solved"
    assert np.max(np.abs(r.x - x_star)) < _ERR
    assert np.array_equal(r.x > 1e-9, x_star > 0)


def test_baselines_agree_with_nncg_bound_only() -> None:
    """All bound-only solvers land on the same minimiser as ``solve_nnqp``."""
    pytest.importorskip("osqp")
    pytest.importorskip("clarabel")
    a, b, _, _ = make_problem(80, 1e4, seed=3)
    op = DenseOperator(a)
    ref = solve_nnqp(op, b).x
    for solve in (bl.solve_osqp, bl.solve_clarabel, bl.solve_lawson_hanson):
        assert np.max(np.abs(solve(op, b).x - ref)) < _ERR


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_duchi_recovers_simplex_plant(seed: int) -> None:
    """Duchi projected gradient recovers the p=1 simplex plant, feasibly."""
    a, b, beta, x_star, _, _ = make_simplex_problem(50, 1e2, beta=1.0, seed=seed)
    op = DenseOperator(a)
    r = bl.solve_duchi(op, b, beta=beta, tol=1e-10)
    assert r.status == "solved"
    assert abs(r.x.sum() - beta) < 1e-8
    assert np.all(r.x >= -1e-9)
    assert np.max(np.abs(r.x - x_star)) < _ERR


def test_simplex_solvers_agree_on_p1() -> None:
    """nncg_eq, OSQP, Clarabel and Duchi agree on the simplex minimiser."""
    pytest.importorskip("osqp")
    pytest.importorskip("clarabel")
    a, b, beta, x_star, _, _ = make_simplex_problem(60, 1e2, beta=1.0, seed=4)
    op = DenseOperator(a)
    b_eq, c_eq = bl.ones_row(60), np.array([beta])

    ref = solve_nnqp_eq(op, b, b_eq, c_eq).x
    assert np.max(np.abs(ref - x_star)) < _ERR
    assert np.max(np.abs(bl.solve_osqp(op, b, b_eq=b_eq, c_eq=c_eq).x - ref)) < _ERR
    assert np.max(np.abs(bl.solve_clarabel(op, b, b_eq=b_eq, c_eq=c_eq).x - ref)) < _ERR
    assert np.max(np.abs(bl.solve_duchi(op, b, beta=beta, tol=1e-10).x - ref)) < _ERR


def test_osqp_clarabel_multipliers_match_nncg() -> None:
    """The recovered equality multiplier matches ``solve_nnqp_eq`` in sign."""
    pytest.importorskip("osqp")
    pytest.importorskip("clarabel")
    a, b, beta, _, lam_star, _ = make_simplex_problem(60, 1e2, beta=1.0, seed=5)
    op = DenseOperator(a)
    b_eq, c_eq = bl.ones_row(60), np.array([beta])
    lam_ref = float(solve_nnqp_eq(op, b, b_eq, c_eq).lam[0])
    assert abs(lam_ref - lam_star) < _ERR
    assert abs(float(bl.solve_osqp(op, b, b_eq=b_eq, c_eq=c_eq).lam[0]) - lam_ref) < 1e-4
    assert abs(float(bl.solve_clarabel(op, b, b_eq=b_eq, c_eq=c_eq).lam[0]) - lam_ref) < 1e-4
