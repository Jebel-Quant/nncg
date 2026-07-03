"""Planted-optimum problem generators.

Synthetic SPD test families with a known optimum in closed form: a
complementary pair ``(x*, s*)`` is planted and ``b = A x* - s*``, so
``(x*, s*)`` solves the LCP and ``x*`` is the unique minimiser. These power
the package's test suite and the numerical study of the accompanying paper —
planting a known optimum is the honest way to test any bound-constrained QP
solver. They live outside the installed package, next to the tests, and stay
importable from here for later experiments and notebooks.
"""

from __future__ import annotations

import numpy as np
from cvx.linalg import Matrix, Vector


def make_problem(
    n: int,
    kappa: float,
    support_frac: float = 0.5,
    seed: int = 0,
) -> tuple[Matrix, Vector, Vector, Vector]:
    """Random SPD problem with prescribed condition number and planted optimum.

    ``A = Q diag(eig) Q^T`` with Haar-random ``Q`` and a geometric spectrum on
    ``[1, kappa]``. A support of size ``round(support_frac * n)`` is chosen;
    ``x*`` is positive there and zero elsewhere, ``s*`` is zero there and
    positive elsewhere, and ``b = A x* - s*``.

    Args:
        n: Problem dimension.
        kappa: Spectral condition number of ``A``.
        support_frac: Fraction of indices in the optimal support.
        seed: Seed of the random generator.

    Returns:
        The tuple ``(A, b, x_star, s_star)``; ``x_star`` is the unique
        minimiser of ``min_{x>=0} 1/2 x^T A x - b^T x`` and ``s_star`` its
        reduced gradient.
    """
    rng = np.random.default_rng(seed)
    eig = np.geomspace(1.0, kappa, n)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = (q * eig) @ q.T
    a = 0.5 * (a + a.T)

    k = max(1, round(support_frac * n))
    perm = rng.permutation(n)
    supp = perm[:k]

    x_star = np.zeros(n)
    x_star[supp] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    off = perm[k:]
    s_star[off] = rng.uniform(0.5, 1.5, size=n - k)
    b = a @ x_star - s_star
    return a, b, x_star, s_star


def make_eq_problem(
    n: int,
    kappa: float,
    p: int,
    support_frac: float = 0.5,
    seed: int = 0,
) -> tuple[Matrix, Vector, Matrix, Vector, Vector, Vector, Vector]:
    """Equality-augmented planted problem for ``min f(x)`` s.t. ``Bx = c``, ``x >= 0``.

    Plants ``(x*, lambda*, s*)`` satisfying the KKT system: ``x* > 0`` on a
    support of size at least ``p + 1`` (so ``B_F`` has full row rank
    generically), ``s* = 0`` there and positive off it, ``lambda*`` arbitrary;
    then ``b = A x* - B^T lambda* - s*`` and ``c = B x*``.

    Args:
        n: Problem dimension.
        kappa: Spectral condition number of ``A``.
        p: Number of equality constraints (rows of ``B``).
        support_frac: Fraction of indices in the optimal support.
        seed: Seed of the random generator.

    Returns:
        The tuple ``(A, b, B, c, x_star, lam_star, s_star)``.
    """
    rng = np.random.default_rng(seed)
    eig = np.geomspace(1.0, kappa, n)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = (q * eig) @ q.T
    a = 0.5 * (a + a.T)
    k = max(p + 1, round(support_frac * n))
    perm = rng.permutation(n)
    supp, off = perm[:k], perm[k:]
    x_star = np.zeros(n)
    x_star[supp] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    s_star[off] = rng.uniform(0.5, 1.5, size=n - k)
    b_eq = rng.standard_normal((p, n))
    lam_star = rng.standard_normal(p)
    b = a @ x_star - b_eq.T @ lam_star - s_star
    c_eq = b_eq @ x_star
    return a, b, b_eq, c_eq, x_star, lam_star, s_star


def make_adversarial(
    n: int,
    seed: int = 0,
    noise: float = 1e-2,
    ridge: float = 1e-6,
) -> tuple[Matrix, Vector]:
    """Anti-correlated design on which the unguarded batch path cycles.

    Columns arrive in near-anti-parallel pairs, ``M = [M0, -M0 + noise * E]``,
    so pushing a variable to its bound flips the sign of its partner and a
    batch exchange systematically over-shoots. Pure block principal pivoting
    (no fallback) cycles on a sizeable fraction of seeds; the guarded loop
    terminates on all of them. The ridge keeps ``A`` a P-matrix.

    Args:
        n: Problem dimension (must be even).
        seed: Seed of the random generator.
        noise: Size of the perturbation breaking exact anti-parallelism.
        ridge: Diagonal shift keeping ``A`` strictly positive definite.

    Returns:
        The pair ``(A, b)``. No optimum is planted; certify a solve with
        :func:`nncg.solver.kkt_violation`.
    """
    rng = np.random.default_rng(seed)
    m0 = rng.standard_normal((n, n // 2))
    m = np.hstack([m0, -m0 + noise * rng.standard_normal((n, n // 2))])
    a = m.T @ m + ridge * np.eye(n)
    return a, m.T @ rng.standard_normal(n)


def make_scaled_problem(
    n: int,
    kappa_core: float,
    spread: float,
    support_frac: float = 0.5,
    seed: int = 0,
) -> tuple[Matrix, Vector, Vector]:
    """Well-conditioned core under a bad diagonal scaling, with planted optimum.

    ``A = D^{1/2} (Q Lambda Q^T) D^{1/2}`` with the entries of ``D`` spread
    over ``[1, spread]``. Jacobi preconditioning removes the scaling, so PCG
    runs at the core's condition number regardless of the spread while plain
    CG pays for it in full.

    Args:
        n: Problem dimension.
        kappa_core: Condition number of the unscaled core.
        spread: Ratio between the largest and smallest diagonal scale.
        support_frac: Fraction of indices in the optimal support.
        seed: Seed of the random generator.

    Returns:
        The tuple ``(A, b, x_star)``.
    """
    rng = np.random.default_rng(seed)
    eig = np.geomspace(1.0, kappa_core, n)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    core = 0.5 * ((q * eig) @ q.T + ((q * eig) @ q.T).T)
    d = rng.permutation(np.geomspace(1.0, spread, n))
    a = core * np.sqrt(np.outer(d, d))
    k = max(1, round(support_frac * n))
    perm = rng.permutation(n)
    x_star = np.zeros(n)
    x_star[perm[:k]] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
    return a, a @ x_star - s_star, x_star
