"""Tests of :mod:`nncg.preconditioners`: the free-block matvec factory and preconditioner builders."""

import numpy as np
import pytest
from cvx.linalg import DenseOperator

from nncg.krylov import KrylovConfig, pcg
from nncg.preconditioners import NystromConfig, _free_matvec, _global_nystrom_sketch, _jacobi, _masked_nystrom, _nystrom
from tests.problems import make_problem
from tests.test_nncg.test_solver import _NoDiagOperator


def _dense_pair(seed: int = 0) -> tuple[DenseOperator, np.ndarray]:
    """Return a matched ``(DenseOperator, ndarray)`` SPD pair for a given seed."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((6, 6))
    a = a @ a.T + np.eye(6)
    return DenseOperator(a), a


def _spd_with_spectrum(eigenvalues: np.ndarray, seed: int) -> np.ndarray:
    """Build an SPD matrix with a prescribed spectrum and a random eigenbasis."""
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.standard_normal((eigenvalues.size, eigenvalues.size)))[0]
    return (q * eigenvalues) @ q.T


# --------------------------------------------------------------------------
# Free-set restriction in the inner matvec factory
# --------------------------------------------------------------------------


def test_free_matvec_uses_restricted() -> None:
    """The callable is the operator's pre-sliced free-block ``matvec``."""
    op, a = _dense_pair()
    idx = np.array([0, 2, 5])
    mv = _free_matvec(op, idx)
    v = np.array([1.0, -2.0, 0.5])
    assert np.allclose(mv(v), a[np.ix_(idx, idx)] @ v)
    # the hoisted path is the restricted operator's matvec, never re-sliced per call
    assert mv.__name__ == "matvec"


# --------------------------------------------------------------------------
# Jacobi preconditioner
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Nyström preconditioner
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Global Nyström sketch, masked per free block
# --------------------------------------------------------------------------


def test_global_nystrom_masks_free_set_exactly() -> None:
    """A sketch built once on the full operator, masked to ``idx``, solves that free block exactly."""
    a = _spd_with_spectrum(np.concatenate([[1e6, 1e5, 1e4], np.geomspace(1.0, 1e2, 37)]), seed=9)
    op = DenseOperator(a)
    sketch = _global_nystrom_sketch(op, NystromConfig(rank=3, seed=0))
    idx = np.arange(0, 40, 2)  # a 20-variable free block
    a_ff = a[np.ix_(idx, idx)]
    rhs = np.arange(1.0, idx.size + 1.0)
    x, _ = pcg(lambda v: a_ff @ v, rhs, KrylovConfig(precond=_masked_nystrom(sketch, idx), tol=1e-8))
    np.testing.assert_allclose(x, np.linalg.solve(a_ff, rhs), rtol=1e-6, atol=1e-8)


def test_global_nystrom_one_sketch_serves_several_free_blocks() -> None:
    """The same global sketch, masked differently, preconditions two distinct free blocks correctly."""
    a = _spd_with_spectrum(np.concatenate([[1e7, 1e6, 1e5], np.geomspace(1.0, 1e2, 47)]), seed=11)
    op = DenseOperator(a)
    sketch = _global_nystrom_sketch(op, NystromConfig(rank=3, seed=0))
    for idx in (np.arange(0, 30), np.arange(20, 50)):
        a_ff = a[np.ix_(idx, idx)]
        rhs = np.arange(1.0, idx.size + 1.0)
        x, _ = pcg(lambda v, a_ff=a_ff: a_ff @ v, rhs, KrylovConfig(precond=_masked_nystrom(sketch, idx), tol=1e-8))
        np.testing.assert_allclose(x, np.linalg.solve(a_ff, rhs), rtol=1e-6, atol=1e-8)


def test_global_nystrom_beats_plain_cg_on_low_rank_spectrum() -> None:
    """A global sketch's masked restriction deflates the free block as effectively as a fresh one.

    Same spectrum and free block as :func:`test_nystrom_beats_plain_cg_on_low_rank_spectrum`,
    but preconditioned from a full-operator sketch masked to ``idx``, not resketched on
    the block itself.
    """
    a = _spd_with_spectrum(np.concatenate([[1e8, 1e7, 1e6], np.geomspace(1.0, 1e2, 97)]), seed=4)
    idx = np.arange(a.shape[0])
    a_ff = a[np.ix_(idx, idx)]
    rhs = np.arange(1.0, idx.size + 1.0)
    _, it_cg = pcg(lambda v: a_ff @ v, rhs, KrylovConfig(tol=1e-8))
    sketch = _global_nystrom_sketch(DenseOperator(a), NystromConfig(rank=3, seed=0))
    _, it_gn = pcg(lambda v: a_ff @ v, rhs, KrylovConfig(precond=_masked_nystrom(sketch, idx), tol=1e-8))
    assert it_gn < it_cg / 2


def test_global_nystrom_rejects_rank_above_numerical_rank() -> None:
    """A sketch wider than the operator's rank captures a negligible eigenvalue."""
    rng = np.random.default_rng(7)
    m = rng.standard_normal((8, 2))  # A = M M^T has rank 2
    with pytest.raises(ValueError, match="negligible eigenvalue"):
        _global_nystrom_sketch(DenseOperator(m @ m.T), config=NystromConfig(rank=5, seed=0))


def test_global_nystrom_rejects_non_positive_shift() -> None:
    """An explicit non-positive shift is refused."""
    a = _spd_with_spectrum(np.geomspace(1.0, 1e3, 10), seed=8)
    with pytest.raises(ValueError, match="shift must be positive"):
        _global_nystrom_sketch(DenseOperator(a), config=NystromConfig(rank=3, shift=-1.0, seed=0))
