"""The numerical study of the paper's Section 7, as regression tests.

Four claims of Section 7 that ``tests.test_nncg`` does not already pin:

- Result (ii): across ``kappa in [10, 1e6]`` the outer count stays single-digit
  and the Bland fallback is dormant on generic data.
- Result (i) / Prop. 3.1: the total CG inner count grows *sublinearly* in
  ``kappa`` — under the ``O(sqrt(kappa))`` worst-case envelope.
- Rank-deficiency result: on a numerically singular Gram operator the split
  ``alpha = 0`` fails to certify a solution, while any ``alpha > 0`` restores
  the ``P``-matrix property and the loop terminates at the planted optimum with
  a KKT certificate — "without the split, termination is luck".
- Matrix-free-at-scale result: a separable Gaussian-blur Gram operator
  ``A = B^T B`` is solved through ``v -> B^T(B v)`` alone (``A`` never formed),
  recovering a non-negative scene and pinning the background at the bound.
"""

import numpy as np
from cvx.linalg import DenseOperator, GramOperator

from nncg import CG, ActiveSetConfig, ActiveSetSolver, kkt_violation
from nncg.krylov import KrylovConfig
from tests.problems import make_problem

KAPPAS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]


def test_outer_count_single_digit_and_fallback_dormant_across_kappa() -> None:
    """Result (ii): outer count single-digit and fallback dormant over kappa in [10, 1e6]."""
    for kappa in KAPPAS:
        for seed in range(3):
            a, b, x_star, _ = make_problem(120, kappa, seed=seed)
            res = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
            assert res.converged
            assert res.outer < 10  # single digit
            assert res.fallback == 0  # Bland fallback never entered on generic data
            assert np.max(np.abs(res.x - x_star)) < 1e-6


def test_inner_count_under_sqrt_kappa_envelope() -> None:
    """Result (i): total CG inner count grows sublinearly in kappa, under the sqrt envelope."""

    def mean_inner(kappa: float) -> float:
        """Mean total CG inner iterations over five seeds at the given kappa."""
        counts = []
        for s in range(5):
            a, b, _, _ = make_problem(120, kappa, seed=s)
            counts.append(ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b).inner)
        return float(np.mean(counts))

    kappa_lo, kappa_hi = 1e2, 1e6
    growth = mean_inner(kappa_hi) / mean_inner(kappa_lo)
    envelope = np.sqrt(kappa_hi / kappa_lo)  # the O(sqrt(kappa)) worst case
    assert growth < envelope


def test_regularisation_buys_the_termination_guarantee() -> None:
    """Rank-deficiency result: alpha=0 fails to certify; any alpha>0 recovers with a KKT certificate.

    ``A = M^T M`` with ``M`` in ``R^{30 x 60}`` has rank <= 30, so both the
    initial free block (size 60) and the optimal one (support 40) are singular:
    at ``alpha = 0`` the loop cannot drive the inner residual down and never
    certifies. The ridge split ``A_alpha = M^T M + alpha I`` restores
    ``A_alpha > 0`` and the loop recovers the planted optimum.
    """
    rng = np.random.default_rng(0)
    m_rows, n, k = 30, 60, 40  # k > m_rows: the optimal free block is singular
    m_mat = rng.standard_normal((m_rows, n))
    perm = rng.permutation(n)
    x_star = np.zeros(n)
    x_star[perm[:k]] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)

    capped = CG(krylov=KrylovConfig(tol=1e-10, maxit=2000))
    config = ActiveSetConfig(max_outer=30)

    # alpha = 0: singular Gram operator, no certificate available. CG either
    # stalls at the iteration cap (uncertified) or breaks down on the singular
    # free block (p @ Ap = 0); both are "no certificate", the paper's point.
    b0 = m_mat.T @ (m_mat @ x_star) - s_star
    op0 = GramOperator(m_mat, ridge=0.0)
    try:
        res0 = ActiveSetSolver(inner=capped, config=config).solve(op0, b0)
        certified_0 = res0.converged and kkt_violation(op0, b0, res0.x) < 1e-6
    except ZeroDivisionError:
        certified_0 = False
    assert not certified_0  # without the split, termination is luck

    # any alpha > 0 restores the P-matrix property and recovers the optimum.
    for alpha in (0.05, 0.2):
        op = GramOperator(m_mat, ridge=alpha)
        b = m_mat.T @ (m_mat @ x_star) + alpha * x_star - s_star
        res = ActiveSetSolver(inner=CG()).solve(op, b)
        assert res.converged
        assert np.max(np.abs(res.x - x_star)) < 1e-6
        assert kkt_violation(op, b, res.x) < 1e-6


def _gaussian_blur_1d(n_side: int, sigma: float) -> np.ndarray:
    """Return a row-normalised ``n_side x n_side`` 1-D Gaussian blur matrix."""
    idx = np.arange(n_side)
    k = np.exp(-((idx[:, None] - idx[None, :]) ** 2) / (2.0 * sigma**2))
    return k / k.sum(axis=1, keepdims=True)


def test_matrix_free_deblurring_recovers_positive_scene() -> None:
    """Matrix-free result: a separable-blur Gram operator recovers a scene, background pinned.

    The blur ``B = K (x) K`` acts on a flattened ``N x N`` image; the Gram
    operator ``A = B^T B + alpha I`` is reached only through ``v -> B^T(B v)``,
    never forming the ``n x n = 36 x 36`` matrix. A planted non-negative scene
    with a zero background is recovered exactly, its background pinned at the
    non-negativity bound.
    """
    n_side = 6
    k = _gaussian_blur_1d(n_side, sigma=1.0)
    b_mat = np.kron(k, k)  # separable blur on the flattened image
    n = n_side * n_side
    alpha = 1e-2  # ridge split for the P-matrix property

    scene = np.zeros(n).reshape(n_side, n_side)
    scene[2:4, 2:4] = np.array([[1.0, 0.8], [0.6, 1.2]])  # a bright block, zero background
    x_star = scene.ravel()
    support = x_star > 0
    s_star = np.zeros(n)
    s_star[~support] = np.random.default_rng(0).uniform(0.5, 1.5, size=int((~support).sum()))

    op = GramOperator(b_mat, ridge=alpha)
    b = b_mat.T @ (b_mat @ x_star) + alpha * x_star - s_star
    res = ActiveSetSolver(inner=CG()).solve(op, b)

    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert kkt_violation(op, b, res.x) < 1e-6
    assert np.array_equal(res.free, support)  # background recovered as active constraints
    assert int((~res.free).sum()) == int((~support).sum())  # every background pixel pinned
