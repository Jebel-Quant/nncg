"""Tests of the CG and Jacobi-PCG inner solvers."""

import numpy as np
import pytest

from nncg.krylov import KrylovConfig, pcg
from tests.problems import make_problem, make_scaled_problem


@pytest.fixture
def spd_system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small SPD system with its exact solution."""
    a, b, _, _ = make_problem(40, 1e3, seed=1)
    return a, b, np.linalg.solve(a, b)


def test_cg_solves_spd(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """CG reaches the exact solution of an SPD system."""
    a, b, x_exact = spd_system
    x, it = pcg(lambda v: a @ v, b, KrylovConfig(tol=1e-12))
    assert it > 0
    assert np.allclose(x, x_exact, atol=1e-7)


def test_cg_zero_rhs() -> None:
    """A zero right-hand side is solved in zero iterations."""
    a, _, _, _ = make_problem(10, 10.0, seed=0)
    x, it = pcg(lambda v: a @ v, np.zeros(10))
    assert it == 0
    assert np.all(x == 0.0)


def test_cg_maxit_returns_iterate(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Hitting the iteration cap returns the current iterate, not garbage."""
    a, b, x_exact = spd_system
    x, it = pcg(lambda v: a @ v, b, KrylovConfig(tol=1e-14, maxit=3))
    assert it == 3
    assert np.linalg.norm(x - x_exact) < np.linalg.norm(x_exact)


def test_cg_warm_start_reduces_iterations(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """A warm start near the solution needs fewer iterations than a cold start."""
    a, b, x_exact = spd_system
    _, it_cold = pcg(lambda v: a @ v, b, KrylovConfig(tol=1e-10))
    _, it_warm = pcg(lambda v: a @ v, b, KrylovConfig(tol=1e-10, x0=x_exact + 1e-6))
    assert it_warm < it_cold


def test_cg_exact_warm_start(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """A warm start that already solves the system returns it in zero iterations.

    The initial residual is then zero, so guarding this case avoids a 0/0 in
    the alpha step.
    """
    a, b, x_exact = spd_system
    x, it = pcg(lambda v: a @ v, b, KrylovConfig(x0=x_exact))
    assert it == 0
    assert np.allclose(x, x_exact)


def test_pcg_solves_spd(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Jacobi PCG reaches the exact solution of an SPD system."""
    a, b, x_exact = spd_system
    x, it = pcg(lambda v: a @ v, b, KrylovConfig(precond=lambda r: r / np.diag(a), tol=1e-12))
    assert it > 0
    assert np.allclose(x, x_exact, atol=1e-7)


def test_pcg_zero_rhs() -> None:
    """A zero right-hand side is solved in zero iterations."""
    a, _, _, _ = make_problem(10, 10.0, seed=0)
    x, it = pcg(lambda v: a @ v, np.zeros(10), KrylovConfig(precond=lambda r: r / np.diag(a)))
    assert it == 0
    assert np.all(x == 0.0)


def test_pcg_maxit_returns_iterate() -> None:
    """Hitting the iteration cap returns the current iterate."""
    a, b, _ = make_scaled_problem(40, 100.0, 1e4, seed=0)
    _, it = pcg(lambda v: a @ v, b, KrylovConfig(precond=lambda r: r / np.diag(a), tol=1e-14, maxit=2))
    assert it == 2


def test_pcg_warm_start_reduces_iterations(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """A warm start near the solution needs fewer PCG iterations than a cold start."""
    a, b, x_exact = spd_system
    _, it_cold = pcg(lambda v: a @ v, b, KrylovConfig(precond=lambda r: r / np.diag(a), tol=1e-10))
    _, it_warm = pcg(lambda v: a @ v, b, KrylovConfig(precond=lambda r: r / np.diag(a), tol=1e-10, x0=x_exact + 1e-6))
    assert it_warm < it_cold


def test_pcg_exact_warm_start(spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """A warm start that already solves the system returns it in zero iterations."""
    a, b, x_exact = spd_system
    x, it = pcg(lambda v: a @ v, b, KrylovConfig(precond=lambda r: r / np.diag(a), x0=x_exact))
    assert it == 0
    assert np.allclose(x, x_exact)


def test_pcg_beats_cg_on_scaled_operator() -> None:
    """Jacobi PCG removes a diagonal scaling that plain CG pays for."""
    a, b, _ = make_scaled_problem(60, 50.0, 1e4, seed=0)
    _, it_cg = pcg(lambda v: a @ v, b, KrylovConfig(tol=1e-10))
    _, it_pcg = pcg(lambda v: a @ v, b, KrylovConfig(precond=lambda r: r / np.diag(a), tol=1e-10))
    assert it_pcg < it_cg / 2
