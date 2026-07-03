"""Tests of the CG and Jacobi-PCG inner solvers."""

import numpy as np
import pytest
from tests.problems import make_problem, make_scaled_problem

from nncg import cg, pcg


@pytest.fixture
def spd_system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small SPD system with its exact solution."""
    a, b, _, _ = make_problem(40, 1e3, seed=1)
    return a, b, np.linalg.solve(a, b)


def test_cg_solves_spd(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """CG reaches the exact solution of an SPD system."""
    a, b, x_exact = spd_system
    x, it = cg(lambda v: a @ v, b, tol=1e-12)
    assert it > 0
    assert np.allclose(x, x_exact, atol=1e-7)


def test_cg_zero_rhs() -> None:
    """A zero right-hand side is solved in zero iterations."""
    a, _, _, _ = make_problem(10, 10.0, seed=0)
    x, it = cg(lambda v: a @ v, np.zeros(10))
    assert it == 0
    assert np.all(x == 0.0)


def test_cg_maxit_returns_iterate(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Hitting the iteration cap returns the current iterate, not garbage."""
    a, b, x_exact = spd_system
    x, it = cg(lambda v: a @ v, b, tol=1e-14, maxit=3)
    assert it == 3
    assert np.linalg.norm(x - x_exact) < np.linalg.norm(x_exact)


def test_cg_warm_start_reduces_iterations(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """A warm start near the solution needs fewer iterations than a cold start."""
    a, b, x_exact = spd_system
    _, it_cold = cg(lambda v: a @ v, b, tol=1e-10)
    _, it_warm = cg(lambda v: a @ v, b, tol=1e-10, x0=x_exact + 1e-6)
    assert it_warm < it_cold


def test_pcg_solves_spd(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Jacobi PCG reaches the exact solution of an SPD system."""
    a, b, x_exact = spd_system
    x, it = pcg(lambda v: a @ v, b, 1.0 / np.diag(a), tol=1e-12)
    assert it > 0
    assert np.allclose(x, x_exact, atol=1e-7)


def test_pcg_zero_rhs() -> None:
    """A zero right-hand side is solved in zero iterations."""
    a, _, _, _ = make_problem(10, 10.0, seed=0)
    x, it = pcg(lambda v: a @ v, np.zeros(10), 1.0 / np.diag(a))
    assert it == 0
    assert np.all(x == 0.0)


def test_pcg_maxit_returns_iterate() -> None:
    """Hitting the iteration cap returns the current iterate."""
    a, b, _ = make_scaled_problem(40, 100.0, 1e4, seed=0)
    _, it = pcg(lambda v: a @ v, b, 1.0 / np.diag(a), tol=1e-14, maxit=2)
    assert it == 2


def test_pcg_beats_cg_on_scaled_operator() -> None:
    """Jacobi PCG removes a diagonal scaling that plain CG pays for."""
    a, b, _ = make_scaled_problem(60, 50.0, 1e4, seed=0)
    _, it_cg = cg(lambda v: a @ v, b, tol=1e-10)
    _, it_pcg = pcg(lambda v: a @ v, b, 1.0 / np.diag(a), tol=1e-10)
    assert it_pcg < it_cg / 2
