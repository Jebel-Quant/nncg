"""Schur-complement elimination of the equality multipliers (paper Section 3).

The equality-augmented free-set solve is the indefinite saddle system
``[[A_F, -B_F^T], [B_F, 0]] [x_F; lambda] = [b_F; c]`` (eq. 3.4). The solver
eliminates ``lambda`` in closed form through ``p + 1`` SPD solves and a
``p x p`` Schur system rather than factorising the indefinite matrix. These
tests pin that the elimination reproduces the saddle solution, and the
pure-normalisation closed form ``x_F = v1 / (1^T v1)`` (Sec. 3, eq. after 3.6).
"""

import numpy as np
from cvx.linalg import DenseOperator

from nncg import CG, ActiveSetSolver
from tests.problems import make_eq_problem, make_problem


def test_eq_multiplier_matches_saddle_point_solve() -> None:
    """Sec. 3: (x_F, lambda) from the Schur elimination solve the saddle system."""
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(80, 1e3, p=3, seed=1)
    res = ActiveSetSolver(inner=CG()).solve_eq(DenseOperator(a), b, b_eq, c_eq)
    assert res.converged
    assert res.lam is not None

    idx = np.flatnonzero(res.free)
    p = b_eq.shape[0]
    a_ff = a[np.ix_(idx, idx)]
    b_f = b_eq[:, idx]
    # Assemble and solve the indefinite saddle system directly.
    kkt = np.block([[a_ff, -b_f.T], [b_f, np.zeros((p, p))]])
    sol = np.linalg.solve(kkt, np.concatenate([b[idx], c_eq]))
    x_f_saddle, lam_saddle = sol[: idx.size], sol[idx.size :]

    assert np.max(np.abs(res.x[idx] - x_f_saddle)) < 1e-6
    assert np.max(np.abs(res.lam - lam_saddle)) < 1e-6


def test_pure_normalisation_closed_form() -> None:
    """Sec. 3: with b=0 and 1^T x = 1 the optimum is ``x_F = v1 / (1^T v1)``, v1 = A_F^{-1} 1."""
    a, _, _, _ = make_problem(60, 1e3, seed=0)
    n = a.shape[0]
    res = ActiveSetSolver(inner=CG()).solve_eq(DenseOperator(a), np.zeros(n), np.ones((1, n)), np.array([1.0]))
    assert res.converged
    assert res.lam is not None

    idx = np.flatnonzero(res.free)
    v1 = np.linalg.solve(a[np.ix_(idx, idx)], np.ones(idx.size))
    denom = float(v1.sum())  # 1^T A_F^{-1} 1 > 0 since A_F is SPD
    assert denom > 0.0

    assert np.max(np.abs(res.x[idx] - v1 / denom)) < 1e-6  # closed form on the free set
    assert float(np.min(res.x)) >= -1e-9  # feasible
    assert abs(float(res.x.sum()) - 1.0) < 1e-9  # meets the normalisation
    assert abs(float(res.lam[0]) - 1.0 / denom) < 1e-6  # scalar multiplier (b=0, v0=0)
