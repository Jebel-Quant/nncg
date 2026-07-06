"""Tests of the preconditioner builders in :mod:`nncg.inner`."""

import numpy as np
import pytest
from cvx.linalg import DenseOperator

from nncg.inner import NystromConfig, _jacobi, _nystrom
from nncg.krylov import KrylovConfig, pcg
from tests.problems import make_problem
from tests.test_operators import _NoDiagOperator


def _spd_with_spectrum(eigenvalues: np.ndarray, seed: int) -> np.ndarray:
    """Build an SPD matrix with a prescribed spectrum and a random eigenbasis."""
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.standard_normal((eigenvalues.size, eigenvalues.size)))[0]
    return (q * eigenvalues) @ q.T


def test_jacobi_applies_reciprocal_of_diag() -> None:
    """``_jacobi(op)`` applies the elementwise product ``(1 / diag(A)) * r``."""
    a = np.diag([1.0, 4.0, 9.0]) + np.array([[0, 0.1, 0], [0.1, 0, 0], [0, 0, 0]])
    r = np.array([2.0, 8.0, 27.0])
    np.testing.assert_allclose(_jacobi(DenseOperator(a))(r), [2.0, 2.0, 3.0])


def test_jacobi_rejects_non_positive_diagonal() -> None:
    """A non-positive diagonal entry is not SPD, so the reciprocal is refused."""
    a = np.diag([1.0, -2.0, 3.0])
    with pytest.raises(ValueError, match="index 1"):
        _jacobi(DenseOperator(a))


def test_jacobi_rejects_non_finite_diagonal() -> None:
    """A non-finite diagonal entry is refused rather than propagated as an inf."""
    a = np.diag([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="index 1"):
        _jacobi(DenseOperator(a))


def test_jacobi_propagates_missing_diag() -> None:
    """A backend without a cheap diagonal propagates ``NotImplementedError``."""
    a, _, _, _ = make_problem(10, 1e2, seed=0)
    with pytest.raises(NotImplementedError, match="diagonal"):
        _jacobi(_NoDiagOperator(a))


def test_jacobi_restricts_to_free_set() -> None:
    """``_jacobi(op, idx)`` is the free-block Jacobi scaling of exactly ``len(idx)``."""
    a, _, _, _ = make_problem(40, 1e3, seed=2)
    op = DenseOperator(a)
    idx = np.array([0, 3, 7, 11])
    r = np.arange(1.0, idx.size + 1.0)
    np.testing.assert_allclose(_jacobi(op, idx)(r), r / np.diag(a)[idx])


def test_nystrom_solves_spd_system() -> None:
    """Nystrom-preconditioned CG reaches the exact solution of an SPD system."""
    a = _spd_with_spectrum(np.concatenate([[1e4, 1e3, 1e2], np.ones(37)]), seed=3)
    b = np.arange(1.0, a.shape[0] + 1.0)
    x, _ = pcg(
        lambda v: a @ v,
        b,
        KrylovConfig(precond=_nystrom(DenseOperator(a), config=NystromConfig(rank=3, seed=0)), tol=1e-10),
    )
    np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-7, atol=1e-9)


def test_nystrom_beats_plain_cg_on_low_rank_spectrum() -> None:
    """Capturing the few dominant eigenvalues collapses the condition number.

    Three widely separated top eigenvalues sit above a spread tail: plain CG
    pays for the full ``1e8 / 1`` condition number, while the rank-3 sketch
    deflates the top block and leaves PCG running at the tail's spread.
    """
    a = _spd_with_spectrum(np.concatenate([[1e8, 1e7, 1e6], np.geomspace(1.0, 1e2, 97)]), seed=4)
    b = np.arange(1.0, a.shape[0] + 1.0)
    _, it_cg = pcg(lambda v: a @ v, b, KrylovConfig(tol=1e-8))
    _, it_ny = pcg(
        lambda v: a @ v,
        b,
        KrylovConfig(precond=_nystrom(DenseOperator(a), config=NystromConfig(rank=3, seed=0)), tol=1e-8),
    )
    assert it_ny < it_cg / 2


def test_nystrom_restricted_to_free_set_solves_block() -> None:
    """``_nystrom(op, idx)`` preconditions the free block ``A[F, F]`` of exact size."""
    a = _spd_with_spectrum(np.concatenate([[1e6, 1e5, 1e4], np.geomspace(1.0, 1e2, 37)]), seed=9)
    op = DenseOperator(a)
    idx = np.arange(0, 40, 2)  # a 20-variable free block
    a_ff = a[np.ix_(idx, idx)]
    rhs = np.arange(1.0, idx.size + 1.0)
    x, _ = pcg(
        lambda v: a_ff @ v, rhs, KrylovConfig(precond=_nystrom(op, idx, NystromConfig(rank=3, seed=0)), tol=1e-8)
    )
    np.testing.assert_allclose(x, np.linalg.solve(a_ff, rhs), rtol=1e-6, atol=1e-8)


def test_nystrom_full_rank_is_near_exact() -> None:
    """A sketch of full rank preconditions to almost the identity — one step suffices."""
    a = _spd_with_spectrum(np.geomspace(1.0, 1e5, 20), seed=5)
    b = np.arange(1.0, a.shape[0] + 1.0)
    precond = _nystrom(DenseOperator(a), config=NystromConfig(rank=20, oversample=0, seed=0))
    _, it = pcg(lambda v: a @ v, b, KrylovConfig(precond=precond, tol=1e-8))
    assert it <= 2


def test_nystrom_config_rejects_non_positive_rank() -> None:
    """A non-positive rank is rejected at config construction."""
    with pytest.raises(ValueError, match="rank must be a positive integer"):
        NystromConfig(rank=0)


def test_nystrom_rejects_rank_above_numerical_rank() -> None:
    """A sketch wider than the operator's rank captures a negligible eigenvalue."""
    rng = np.random.default_rng(7)
    m = rng.standard_normal((8, 2))  # A = M M^T has rank 2
    with pytest.raises(ValueError, match="negligible eigenvalue"):
        _nystrom(DenseOperator(m @ m.T), config=NystromConfig(rank=5, seed=0))


def test_nystrom_rejects_non_positive_shift() -> None:
    """An explicit non-positive shift is refused."""
    a = _spd_with_spectrum(np.geomspace(1.0, 1e3, 10), seed=8)
    with pytest.raises(ValueError, match="shift must be positive"):
        _nystrom(DenseOperator(a), config=NystromConfig(rank=3, shift=-1.0, seed=0))
