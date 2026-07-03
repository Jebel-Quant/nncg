"""Tests of the operator interface: ``A`` enters only as a ``SymmetricOperator``.

The solvers accept the quadratic term exclusively as a
``cvx.linalg.SymmetricOperator``, accessed through block products
(``apply_free`` for the CG inner solves, ``matvec`` for the reduced gradient,
``solve_free`` behind ``inner="exact"``). The Gram case makes the paper's
memory claim concrete: with ``A = M^T M + ridge I`` supplied as a
``GramOperator``, the ``n x n`` matrix is never formed.
"""

import numpy as np
import pytest
from cvx.linalg import DenseOperator, GramOperator, SymmetricOperator

from nncg import kkt_violation, make_problem, solve_nnqp, solve_nnqp_eq


class _NoDiagOperator(SymmetricOperator):
    """Dense-backed operator that keeps the base class's diag (which raises)."""

    def __init__(self, a: np.ndarray) -> None:
        """Store the backing array."""
        self._a = a

    @property
    def n(self) -> int:
        """Dimension of the operator."""
        return int(self._a.shape[0])

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Return ``A @ x``."""
        return self._a @ x

    def block_matvec(self, rows: object, cols: object, v: np.ndarray) -> np.ndarray:
        """Return ``A[rows, cols] @ v``."""
        return self._a[np.ix_(rows, cols)] @ v

    def solve_free(self, free: object, rhs: np.ndarray) -> np.ndarray:
        """Solve the free block directly."""
        return np.linalg.solve(self._a[np.ix_(free, free)], rhs)

    def rcond_free(self, free: object) -> float:
        """Reciprocal condition number of the free block."""
        return 1.0 / float(np.linalg.cond(self._a[np.ix_(free, free)]))


def _plant_gram_problem(m_rows: int, n: int, ridge: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plant an optimum for ``A = M^T M + ridge I`` using products with ``M`` only."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((m_rows, n))
    k = n // 2
    perm = rng.permutation(n)
    x_star = np.zeros(n)
    x_star[perm[:k]] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
    b = m.T @ (m @ x_star) + ridge * x_star - s_star
    return m, b, x_star


def test_gram_operator_recovers_planted_optimum() -> None:
    """The Gram-structured solve recovers a planted optimum matrix-free.

    Both the planting and the solve touch ``A = M^T M + ridge I`` only
    through products with ``M`` — the ``n x n`` Gram matrix never exists.
    """
    m, b, x_star = _plant_gram_problem(30, 60, ridge=1.0, seed=0)
    op = GramOperator(m, ridge=1.0)
    res = solve_nnqp(op, b)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert kkt_violation(op, b, res.x) < 1e-6


def test_exact_inner_uses_operator_solve_free() -> None:
    """``inner="exact"`` routes through ``op.solve_free`` for both backends."""
    a, b, x_star, _ = make_problem(60, 1e2, seed=1)
    res = solve_nnqp(DenseOperator(a), b, inner="exact")
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6

    m, b_g, x_star_g = _plant_gram_problem(30, 60, ridge=1.0, seed=2)
    res_g = solve_nnqp(GramOperator(m, ridge=1.0), b_g, inner="exact")
    assert res_g.converged
    assert np.max(np.abs(res_g.x - x_star_g)) < 1e-6


def test_pcg_preconditions_from_operator_diag() -> None:
    """Jacobi PCG reads its preconditioner off ``op.diag`` — no matrix needed."""
    m, b, x_star = _plant_gram_problem(30, 60, ridge=1.0, seed=3)
    res = solve_nnqp(GramOperator(m, ridge=1.0), b, inner="pcg")
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6


def test_pcg_without_diag_raises() -> None:
    """A backend without a cheap diagonal refuses Jacobi PCG."""
    a, b, _, _ = make_problem(30, 1e2, seed=0)
    with pytest.raises(NotImplementedError, match="diagonal"):
        solve_nnqp(_NoDiagOperator(a), b, inner="pcg")


def test_exact_inner_guards_singular_free_block() -> None:
    """``inner="exact"`` refuses a rank-deficient free block via ``rcond_free``.

    An un-ridged Gram operator with fewer rows than free variables has a
    singular free block on the initial all-free set; the guard turns the
    latent factorisation failure into a clean diagnostic.
    """
    rng = np.random.default_rng(0)
    m = rng.standard_normal((10, 30))  # rank 10 < n = 30
    b = m.T @ rng.standard_normal(10)
    with pytest.raises(ValueError, match="singular"):
        solve_nnqp(GramOperator(m, ridge=0.0), b, inner="exact")


def test_dimension_mismatch_is_rejected() -> None:
    """An operator whose dimension disagrees with len(b) is refused at entry."""
    a, b, _, _ = make_problem(20, 1e2, seed=0)
    op = DenseOperator(a)
    b_short = b[:-1]
    with pytest.raises(ValueError, match="dimension"):
        solve_nnqp(op, b_short)
    with pytest.raises(ValueError, match="dimension"):
        solve_nnqp_eq(op, b_short, np.ones((1, 19)), np.array([1.0]))
    with pytest.raises(ValueError, match="dimension"):
        kkt_violation(op, b_short, np.zeros(19))


def test_dense_arrays_are_rejected() -> None:
    """A dense array is refused with a pointer to DenseOperator."""
    a, b, _, _ = make_problem(20, 1e2, seed=0)
    with pytest.raises(TypeError, match="DenseOperator"):
        solve_nnqp(a, b)
    with pytest.raises(TypeError, match="DenseOperator"):
        solve_nnqp_eq(a, b, np.ones((1, 20)), np.array([1.0]))
    with pytest.raises(TypeError, match="DenseOperator"):
        kkt_violation(a, b, np.zeros(20))


def test_eq_solver_accepts_gram_operator() -> None:
    """The equality-augmented loop runs matrix-free on a Gram operator."""
    m, b, x_star = _plant_gram_problem(30, 60, ridge=1.0, seed=4)
    ones = np.ones((1, 60))
    beta = np.array([float(x_star.sum())])
    res = solve_nnqp_eq(GramOperator(m, ridge=1.0), b, ones, beta)
    assert res.converged
    assert res.lam is not None
    assert abs(float(res.x.sum()) - float(beta[0])) < 1e-8
    assert np.max(np.abs(res.x - x_star)) < 1e-6
