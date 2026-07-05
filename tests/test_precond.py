"""Tests of the preconditioner builders in :mod:`nncg.precond`."""

import numpy as np
import pytest
from cvx.linalg import DenseOperator

from nncg.krylov import cg, pcg
from nncg.precond import diagonal, identity, inverse_diagonal, jacobi, nystrom
from tests.problems import make_problem
from tests.test_operators import _NoDiagOperator


def _spd_with_spectrum(eigenvalues: np.ndarray, seed: int) -> np.ndarray:
    """Build an SPD matrix with a prescribed spectrum and a random eigenbasis."""
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.standard_normal((eigenvalues.size, eigenvalues.size)))[0]
    return (q * eigenvalues) @ q.T


def test_inverse_diagonal_is_reciprocal_of_diag() -> None:
    """``inverse_diagonal`` returns ``1 / diag(A)`` read off the operator."""
    a = np.diag([1.0, 4.0, 9.0]) + np.array([[0, 0.1, 0], [0.1, 0, 0], [0, 0, 0]])
    dinv = inverse_diagonal(DenseOperator(a))
    np.testing.assert_allclose(dinv, [1.0, 0.25, 1.0 / 9.0])


def test_inverse_diagonal_rejects_non_positive_diagonal() -> None:
    """A non-positive diagonal entry is not SPD, so the reciprocal is refused."""
    a = np.diag([1.0, -2.0, 3.0])
    with pytest.raises(ValueError, match="index 1"):
        inverse_diagonal(DenseOperator(a))


def test_inverse_diagonal_rejects_non_finite_diagonal() -> None:
    """A non-finite diagonal entry is refused rather than propagated as an inf."""
    a = np.diag([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="index 1"):
        inverse_diagonal(DenseOperator(a))


def test_inverse_diagonal_propagates_missing_diag() -> None:
    """A backend without a cheap diagonal propagates ``NotImplementedError``."""
    a, _, _, _ = make_problem(10, 1e2, seed=0)
    with pytest.raises(NotImplementedError, match="diagonal"):
        inverse_diagonal(_NoDiagOperator(a))


def test_identity_reproduces_plain_cg() -> None:
    """PCG with the identity preconditioner matches plain CG iterate for iterate."""
    a, b, _, _ = make_problem(40, 1e3, seed=1)
    x_cg, it_cg = cg(lambda v: a @ v, b, tol=1e-10)
    x_id, it_id = pcg(lambda v: a @ v, b, identity(), tol=1e-10)
    assert it_id == it_cg
    np.testing.assert_allclose(x_id, x_cg, rtol=1e-10, atol=1e-12)


def test_jacobi_matches_explicit_diagonal() -> None:
    """``jacobi(op)`` is ``diagonal(inverse_diagonal(op))``."""
    a, b, _, _ = make_problem(40, 1e3, seed=2)
    op = DenseOperator(a)
    r = b.copy()
    np.testing.assert_allclose(jacobi(op)(r), diagonal(inverse_diagonal(op))(r))


def test_nystrom_solves_spd_system() -> None:
    """Nystrom-preconditioned CG reaches the exact solution of an SPD system."""
    a = _spd_with_spectrum(np.concatenate([[1e4, 1e3, 1e2], np.ones(37)]), seed=3)
    b = np.arange(1.0, a.shape[0] + 1.0)
    x, _ = pcg(lambda v: a @ v, b, nystrom(lambda v: a @ v, a.shape[0], rank=3, seed=0), tol=1e-10)
    np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-7, atol=1e-9)


def test_nystrom_beats_plain_cg_on_low_rank_spectrum() -> None:
    """Capturing the few dominant eigenvalues collapses the condition number.

    Three widely separated top eigenvalues sit above a spread tail: plain CG
    pays for the full ``1e8 / 1e2`` condition number, while the rank-3 sketch
    deflates the top block and leaves PCG running at the tail's spread.
    """
    a = _spd_with_spectrum(np.concatenate([[1e8, 1e7, 1e6], np.geomspace(1.0, 1e2, 97)]), seed=4)
    b = np.arange(1.0, a.shape[0] + 1.0)
    _, it_cg = cg(lambda v: a @ v, b, tol=1e-8)
    _, it_ny = pcg(lambda v: a @ v, b, nystrom(lambda v: a @ v, a.shape[0], rank=3, seed=0), tol=1e-8)
    assert it_ny < it_cg / 2


def test_nystrom_full_rank_is_near_exact() -> None:
    """A sketch of full rank preconditions to almost the identity — one step suffices."""
    a = _spd_with_spectrum(np.geomspace(1.0, 1e5, 20), seed=5)
    b = np.arange(1.0, a.shape[0] + 1.0)
    _, it = pcg(lambda v: a @ v, b, nystrom(lambda v: a @ v, a.shape[0], rank=20, oversample=0, seed=0), tol=1e-8)
    assert it <= 2


def test_nystrom_rejects_non_positive_rank() -> None:
    """A non-positive rank is a usage error."""
    a = _spd_with_spectrum(np.ones(5), seed=6)
    with pytest.raises(ValueError, match="rank must be"):
        nystrom(lambda v: a @ v, 5, rank=0)


def test_nystrom_rejects_rank_above_numerical_rank() -> None:
    """A sketch wider than the operator's rank captures a zero eigenvalue."""
    rng = np.random.default_rng(7)
    m = rng.standard_normal((8, 2))  # A = M M^T has rank 2
    with pytest.raises(ValueError, match="negligible eigenvalue"):
        nystrom(lambda v: m @ (m.T @ v), 8, rank=5, seed=0)


def test_nystrom_rejects_non_positive_shift() -> None:
    """An explicit non-positive shift is refused."""
    a = _spd_with_spectrum(np.geomspace(1.0, 1e3, 10), seed=8)
    with pytest.raises(ValueError, match="shift must be positive"):
        nystrom(lambda v: a @ v, 10, rank=3, shift=-1.0, seed=0)
