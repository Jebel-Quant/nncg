"""Inner-solve guarantees of the paper's Section 3 / Section 6.

Three quantitative claims about the matrix-free CG inner solver:

- Proposition 3.1 (CG convergence): the energy-norm error contracts under the
  Chebyshev envelope ``||e_k||_A <= 2 rho^k ||e_0||_A`` with
  ``rho = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)``.
- Proposition 6.2 (``prop:pcg``): an SPD preconditioner ``P`` reduces the rate
  to ``kappa_P = kappa(P^{-1} A)``, and is *ideal* (``kappa_P = 1``, one
  iteration) when ``P = A``.
- Lemma 5.2 (inexact inner solves): the iterate stopped at residual ``rho``
  satisfies ``||x~ - x*||_inf <= rho / lambda_min(A)``.
"""

import numpy as np

from nncg.krylov import KrylovConfig, pcg
from tests.problems import make_problem


def _energy_norm(a: np.ndarray, e: np.ndarray) -> float:
    """Return the ``A``-energy norm ``(e^T A e)^{1/2}`` of the error ``e``."""
    return float(np.sqrt(e @ (a @ e)))


def test_cg_energy_norm_within_chebyshev_bound() -> None:
    """Prop. 3.1: every CG iterate stays under the Chebyshev energy-norm envelope."""
    a, _, _, _ = make_problem(50, 1e3, seed=0)
    rng = np.random.default_rng(1)
    b = rng.standard_normal(50)
    x_exact = np.linalg.solve(a, b)

    eig = np.linalg.eigvalsh(a)
    kappa = eig[-1] / eig[0]
    rho = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
    e0_energy = _energy_norm(a, x_exact)  # warm start x0 = 0, so e0 = x_exact

    for k in range(1, 13):
        # tol=0 never fires, so pcg runs exactly k iterations and returns x_k.
        x_k, it = pcg(lambda v: a @ v, b, KrylovConfig(tol=0.0, maxit=k))
        assert it == k
        ek_energy = _energy_norm(a, x_exact - x_k)
        bound = 2.0 * rho**k * e0_energy
        assert ek_energy <= bound * (1.0 + 1e-9)


def test_ideal_preconditioner_solves_in_one_iteration() -> None:
    """Prop. 6.2: the exact preconditioner ``P = A`` gives ``kappa_P = 1``, one PCG step."""
    a, _, _, _ = make_problem(40, 1e4, seed=2)
    rng = np.random.default_rng(3)
    b = rng.standard_normal(40)

    # precond r -> A^{-1} r makes M^{-1} A = I, so kappa_P = 1.
    x, it = pcg(lambda v: a @ v, b, KrylovConfig(precond=lambda r: np.linalg.solve(a, r), tol=1e-8))
    assert it == 1
    assert np.allclose(x, np.linalg.solve(a, b))


def test_inexact_inner_solve_error_bound() -> None:
    """Lemma 5.2: a CG iterate at residual ``rho`` obeys ``||x~ - x*||_inf <= rho / lambda_min``."""
    a, _, _, _ = make_problem(60, 1e4, seed=4)
    rng = np.random.default_rng(5)
    b = rng.standard_normal(60)
    x_exact = np.linalg.solve(a, b)
    lam_min = float(np.linalg.eigvalsh(a)[0])

    # Stop CG early so the residual is genuinely nonzero and the bound bites.
    x_tilde, _ = pcg(lambda v: a @ v, b, KrylovConfig(tol=1e-1))
    residual = float(np.linalg.norm(b - a @ x_tilde))
    assert residual > 0.0
    assert np.max(np.abs(x_tilde - x_exact)) <= residual / lam_min * (1.0 + 1e-9)
