"""The regularising split's conditioning bound (paper Proposition 6.1).

The convex split ``A_alpha = (1 - alpha) A + alpha R^T R`` (eq. 6.1) with
``R^T R = I`` obeys the Weyl bound

    kappa(A_alpha) <= ((1 - alpha) lambda_max(A) + alpha) / alpha
                    = (1 - alpha)/alpha * lambda_max(A) + 1,

which is strictly decreasing in ``alpha`` and approaches ``1`` as
``alpha -> 1``. Since the split is also what restores ``A_alpha > 0`` (the
``P``-matrix property) on a rank-deficient operator, its numerical-recovery
consequence is exercised in ``test_results.test_regularisation_buys_the_termination_guarantee``.
"""

import numpy as np

from tests.problems import make_problem


def _condition_number(a: np.ndarray) -> float:
    """Return the spectral condition number ``lambda_max / lambda_min`` of SPD ``a``."""
    eig = np.linalg.eigvalsh(a)
    return float(eig[-1] / eig[0])


def test_regularising_split_condition_number_bound() -> None:
    """Prop. 6.1: kappa(A_alpha) stays under the Weyl bound, falls with alpha, tends to 1."""
    a, _, _, _ = make_problem(80, 1e5, seed=0)
    n = a.shape[0]
    lam_max = float(np.linalg.eigvalsh(a)[-1])

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    kappas = []
    for alpha in alphas:
        a_alpha = (1.0 - alpha) * a + alpha * np.eye(n)  # R^T R = I
        kappa = _condition_number(a_alpha)
        weyl_bound = ((1.0 - alpha) * lam_max + alpha) / alpha
        assert kappa <= weyl_bound * (1.0 + 1e-9)  # Weyl inequality
        kappas.append(kappa)

    # strictly decreasing in alpha
    assert all(kappas[i + 1] < kappas[i] for i in range(len(kappas) - 1))
    # kappa(A_alpha) -> kappa(R^T R) = 1 as alpha -> 1 (needs alpha close to 1 since
    # kappa(A_alpha) ~= 1 + (1 - alpha) lambda_max here)
    alpha_heavy = 1.0 - 1e-7
    a_heavy = (1.0 - alpha_heavy) * a + alpha_heavy * np.eye(n)
    assert _condition_number(a_heavy) < 1.05
