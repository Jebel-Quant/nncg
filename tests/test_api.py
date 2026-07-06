"""Tests of the one-call convenience wrappers ``solve_nnqp`` / ``solve_nnqp_eq``.

They must be logic-free pass-throughs: the same problem solved through the
wrapper and through ``ActiveSetSolver`` directly must give an identical
``Result``. What is tested here is only the wrapping conveniences — array →
operator, string → inner solver, keyword → ``ActiveSetConfig`` — not the
mathematics, which the solver's own suite covers.
"""

import numpy as np
import pytest
from cvx.linalg import DenseOperator, GramOperator

from nncg import (
    CG,
    ActiveSetConfig,
    ActiveSetSolver,
    Jacobi,
    Nystrom,
    NystromConfig,
    kkt_violation,
    solve_nnqp,
    solve_nnqp_eq,
)
from tests.problems import make_eq_problem, make_problem


def test_solve_nnqp_matches_active_set_solver() -> None:
    """The wrapper is a pure pass-through: same x/outer/inner as the explicit path."""
    a, b, x_star, _ = make_problem(80, 1e4, seed=0)
    res = solve_nnqp(a, b, inner="cg")
    ref = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
    assert res.converged
    np.testing.assert_array_equal(res.x, ref.x)
    assert (res.outer, res.inner, res.fallback) == (ref.outer, ref.inner, ref.fallback)
    assert np.max(np.abs(res.x - x_star)) < 1e-6


def test_solve_nnqp_default_inner_is_cg() -> None:
    """Omitting ``inner`` uses plain CG."""
    a, b, _, _ = make_problem(60, 1e3, seed=1)
    res = solve_nnqp(a, b)
    ref = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
    np.testing.assert_array_equal(res.x, ref.x)


@pytest.mark.parametrize(("name", "instance"), [("cg", CG()), ("jacobi", Jacobi()), ("nystrom", Nystrom())])
def test_inner_string_shortcut_equals_default_instance(name: str, instance: object) -> None:
    """A shortcut string default-constructs the matching inner solver."""
    a, b, _, _ = make_problem(60, 1e3, seed=2)
    res = solve_nnqp(a, b, inner=name)  # type: ignore[arg-type]
    ref = ActiveSetSolver(inner=instance).solve(DenseOperator(a), b)  # type: ignore[arg-type]
    np.testing.assert_array_equal(res.x, ref.x)


def test_inner_instance_is_used_as_is() -> None:
    """A configured inner-solver instance is passed straight through."""
    a, b, _, _ = make_problem(60, 1e3, seed=3)
    inner = Nystrom(nystrom=NystromConfig(rank=20, seed=7))
    res = solve_nnqp(a, b, inner=inner)
    ref = ActiveSetSolver(inner=inner).solve(DenseOperator(a), b)
    np.testing.assert_array_equal(res.x, ref.x)


def test_operator_is_used_unchanged() -> None:
    """A SymmetricOperator is not re-wrapped — the matrix-free path is preserved."""
    rng = np.random.default_rng(0)
    m = rng.standard_normal((40, 25))
    op = GramOperator(m, ridge=1.0)  # A = M'M + I, never formed
    b = rng.standard_normal(25)
    res = solve_nnqp(op, b, inner="cg")
    ref = ActiveSetSolver(inner=CG()).solve(op, b)
    np.testing.assert_array_equal(res.x, ref.x)
    assert kkt_violation(op, b, res.x) < 1e-7 * (1.0 + float(np.linalg.norm(b)))


def test_outer_keywords_reach_active_set_config() -> None:
    """The outer-loop keywords are bundled into ActiveSetConfig and honoured."""
    a, b, _, _ = make_problem(60, 1e3, seed=4)
    res = solve_nnqp(a, b, track=True, max_outer=1)
    ref = ActiveSetSolver(inner=CG(), config=ActiveSetConfig(track=True, max_outer=1)).solve(DenseOperator(a), b)
    assert res.traj is not None  # track honoured
    assert res.outer == 1
    assert not res.converged
    np.testing.assert_array_equal(res.x, ref.x)


def test_unknown_inner_string_raises_valueerror() -> None:
    """A mistyped shortcut is a friendly ValueError, not a raw KeyError."""
    a, b, _, _ = make_problem(10, 10.0, seed=0)
    with pytest.raises(ValueError, match="unknown inner solver"):
        solve_nnqp(a, b, inner="nystorm")  # type: ignore[arg-type]


def test_solve_nnqp_eq_matches_active_set_solver() -> None:
    """The equality wrapper is a pure pass-through of solve_eq."""
    a, b, b_eq, c_eq, x_star, _, _ = make_eq_problem(80, 1e3, p=3, seed=3)
    res = solve_nnqp_eq(a, b, b_eq, c_eq, inner="jacobi")
    ref = ActiveSetSolver(inner=Jacobi()).solve_eq(DenseOperator(a), b, b_eq, c_eq)
    assert res.converged
    np.testing.assert_array_equal(res.x, ref.x)
    assert res.lam is not None
    np.testing.assert_array_equal(res.lam, ref.lam)
    assert np.max(np.abs(res.x - x_star)) < 1e-6
