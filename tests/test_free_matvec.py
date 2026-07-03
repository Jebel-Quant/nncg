"""Tests for the hoisted free-set restriction in the inner matvec factory."""

import numpy as np
from cvx.linalg import DenseOperator, SymmetricOperator

from nncg.solver import _free_matvec


def _dense_pair(seed: int = 0) -> tuple[DenseOperator, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((6, 6))
    a = a @ a.T + np.eye(6)
    return DenseOperator(a), a


def test_free_matvec_uses_restricted_when_available() -> None:
    """With a restricted-capable operator, the callable is the pre-sliced matvec."""
    op, a = _dense_pair()
    idx = np.array([0, 2, 5])
    mv = _free_matvec(op, idx)
    v = np.array([1.0, -2.0, 0.5])
    assert np.allclose(mv(v), a[np.ix_(idx, idx)] @ v)
    if hasattr(op, "restricted"):
        # the hoisted path must not be the apply_free lambda
        assert mv.__name__ == "matvec"


def test_free_matvec_falls_back_without_restricted() -> None:
    """An operator without ``restricted`` still works through ``apply_free``."""
    op, a = _dense_pair(1)
    idx = np.array([1, 3, 4])

    class _Legacy:
        """Operator stub exposing only apply_free."""

        def apply_free(self, free: np.ndarray, v: np.ndarray) -> np.ndarray:
            return a[np.ix_(free, free)] @ v

    mv = _free_matvec(_Legacy(), idx)  # type: ignore[arg-type]
    v = np.array([0.5, 1.5, -1.0])
    assert np.allclose(mv(v), a[np.ix_(idx, idx)] @ v)


def test_free_matvec_falls_back_on_raising_default() -> None:
    """A backend inheriting the raising default of ``restricted`` falls back cleanly."""
    op, a = _dense_pair(2)
    idx = np.array([0, 4])

    class _NoRestricted(SymmetricOperator):
        """Minimal backend leaving the base default ``restricted`` in place."""

        @property
        def n(self) -> int:
            return 6

        def matvec(self, x):  # noqa: ANN001, ANN201
            return a @ x

        def block_matvec(self, rows, cols, v):  # noqa: ANN001, ANN201
            return a[np.ix_(np.asarray(rows), np.asarray(cols))] @ v

        def solve_free(self, free, rhs):  # noqa: ANN001, ANN201
            return np.linalg.solve(a[np.ix_(free, free)], rhs)

        def rcond_free(self, free) -> float:  # noqa: ANN001
            return 1.0

    mv = _free_matvec(_NoRestricted(), idx)
    v = np.array([1.0, 2.0])
    assert np.allclose(mv(v), a[np.ix_(idx, idx)] @ v)
