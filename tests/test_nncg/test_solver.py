"""Tests of the active-set loop: the paper's numerical study as a test suite.

Covers the bound-constrained solve, the equality-augmented variant
(``B x = c`` via the Schur complement), the operator interface (``A`` enters
only as a ``cvx.linalg.SymmetricOperator``), the anti-cycling / Bland fallback
that underwrites finite termination, and the planted-optimum generators the
whole suite is built on.
"""

import numpy as np
import pytest
from cvx.linalg import DenseOperator, GramOperator, SymmetricOperator

from nncg import (
    CG,
    ActiveSetConfig,
    ActiveSetSolver,
    Exact,
    InnerSolver,
    Jacobi,
    Nystrom,
    NystromConfig,
    kkt_violation,
)
from tests.problems import (
    make_adversarial,
    make_eq_problem,
    make_problem,
    make_scaled_eq_problem,
    make_scaled_problem,
)


class _NoDiagOperator(SymmetricOperator):
    """Dense-backed operator that keeps the base class's diag (which raises)."""

    def __init__(self, a: np.ndarray) -> None:
        """Store the backing array."""
        self._a = a

    @property
    def n(self) -> int:
        """Dimension of the operator."""
        return int(self._a.shape[0])

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Return ``A @ x``."""
        return self._a @ x

    def block_matvec(self, rows: object, cols: object, v: np.ndarray) -> np.ndarray:
        """Return ``A[rows, cols] @ v``."""
        return self._a[np.ix_(rows, cols)] @ v

    def solve_free(self, free: object, rhs: np.ndarray) -> np.ndarray:
        """Solve the free block directly."""
        return np.linalg.solve(self._a[np.ix_(free, free)], rhs)

    def rcond_free(self, free: object) -> float:
        """Reciprocal condition number of the free block."""
        return 1.0 / float(np.linalg.cond(self._a[np.ix_(free, free)]))


def _plant_gram_problem(m_rows: int, n: int, ridge: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plant an optimum for ``A = M^T M + ridge I`` using products with ``M`` only."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((m_rows, n))
    k = n // 2
    perm = rng.permutation(n)
    x_star = np.zeros(n)
    x_star[perm[:k]] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
    b = m.T @ (m @ x_star) + ridge * x_star - s_star
    return m, b, x_star


# --------------------------------------------------------------------------
# Bound-constrained active-set loop
# --------------------------------------------------------------------------


def test_active_set_solver_class_api() -> None:
    """``ActiveSetSolver`` bundles config + inner and solves via its methods."""
    a, b, x_star, _ = make_problem(60, 1e3, seed=0)
    solver = ActiveSetSolver(config=ActiveSetConfig(track=True), inner=Jacobi())
    res = solver.solve(DenseOperator(a), b)
    assert res.converged
    assert res.traj is not None  # config.track honoured
    assert np.max(np.abs(res.x - x_star)) < 1e-6


@pytest.mark.parametrize("kappa", [1e1, 1e3, 1e5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_planted_optimum(kappa: float, seed: int) -> None:
    """The loop recovers the planted optimum across condition numbers and seeds."""
    a, b, x_star, _ = make_problem(80, kappa, seed=seed)
    res = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    # the complementarity products scale with ||b||, so the certificate is
    # asserted relative to the problem's magnitude
    assert kkt_violation(DenseOperator(a), b, res.x) < 1e-7 * (1.0 + float(np.linalg.norm(b)))


def test_outer_count_small_and_fallback_dormant() -> None:
    """On generic data the outer count is single-digit and the fallback dormant."""
    for seed in range(3):
        a, b, _, _ = make_problem(80, 1e4, seed=seed)
        res = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
        assert res.outer < 10
        assert res.fallback == 0


def test_free_mask_matches_support() -> None:
    """The final free mask is the support of the planted optimum."""
    a, b, x_star, _ = make_problem(60, 1e2, seed=3)
    res = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
    assert np.array_equal(res.free, x_star > 0)


def test_exact_inner_matches_cg_trajectory() -> None:
    """CG and exact inner solves visit the same free sets (inexactness lemma)."""
    for seed in range(3):
        a, b, _, _ = make_problem(60, 1e4, seed=seed)
        r_cg = ActiveSetSolver(inner=CG(), config=ActiveSetConfig(track=True)).solve(DenseOperator(a), b)
        r_ex = ActiveSetSolver(config=ActiveSetConfig(track=True), inner=Exact()).solve(DenseOperator(a), b)
        assert r_cg.traj == r_ex.traj


def test_pcg_inner_solve() -> None:
    """The Jacobi-PCG inner solver reaches the same optimum, cheaper when scaled."""
    a, b, x_star = make_scaled_problem(80, 50.0, 1e4, seed=1)
    r_cg = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
    r_pcg = ActiveSetSolver(inner=Jacobi()).solve(DenseOperator(a), b)
    assert np.max(np.abs(r_cg.x - x_star)) < 1e-6
    assert np.max(np.abs(r_pcg.x - x_star)) < 1e-6
    # A comfortable margin, not a bare `<`: the exact counts vary with the BLAS
    # backend, but Jacobi removes a 1e4 diagonal spread, so the win is large.
    assert r_pcg.inner <= 0.7 * r_cg.inner


def test_nystrom_inner_recovers_planted_optimum() -> None:
    """The Nyström-preconditioned inner solver reaches the same optimum as CG."""
    a, b, x_star, _ = make_problem(80, 1e4, seed=1)
    op = DenseOperator(a)
    res = ActiveSetSolver(inner=Nystrom(nystrom=NystromConfig(rank=15))).solve(op, b)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert kkt_violation(op, b, res.x) < 1e-7 * (1.0 + float(np.linalg.norm(b)))


def test_nystrom_inner_matches_cg_trajectory() -> None:
    """Nyström preconditioning changes only inner cost, not the visited free sets."""
    for seed in range(3):
        a, b, _, _ = make_problem(60, 1e4, seed=seed)
        op = DenseOperator(a)
        traj_cg = ActiveSetSolver(inner=CG(), config=ActiveSetConfig(track=True)).solve(op, b).traj
        traj_ny = ActiveSetSolver(config=ActiveSetConfig(track=True), inner=Nystrom()).solve(op, b).traj
        assert traj_cg == traj_ny


def test_max_outer_cap_reports_nonconvergence() -> None:
    """An outer-step cap stops the loop with converged=False."""
    a, b, _, _ = make_problem(60, 1e3, seed=0)
    res = ActiveSetSolver(inner=CG(), config=ActiveSetConfig(max_outer=1)).solve(DenseOperator(a), b)
    assert not res.converged
    assert res.outer == 1


def test_warm_start_support_stable_single_step() -> None:
    """Across a support-stable parameter step the warm loop takes one outer step."""
    a, b, _, _ = make_problem(80, 1e3, seed=0)
    first = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
    delta = 1e-4 * np.linalg.norm(b) * np.ones_like(b) / np.sqrt(len(b))
    second_cold = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b + delta)
    assert np.array_equal(second_cold.free, first.free)  # support-stable step
    second_warm = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b + delta, warm=(first.free, first.x))
    assert second_warm.converged
    assert second_warm.outer == 1
    assert second_warm.inner < second_cold.inner
    assert np.max(np.abs(second_warm.x - second_cold.x)) < 1e-6


def test_warm_start_pcg_reduces_inner_iterations() -> None:
    """The PCG inner solver consumes the warm seed: fewer inner iters, one outer step."""
    a, b, _, _ = make_problem(80, 1e3, seed=0)
    op = DenseOperator(a)
    solver = ActiveSetSolver(inner=Jacobi())
    first = solver.solve(op, b)
    delta = 1e-4 * np.linalg.norm(b) * np.ones_like(b) / np.sqrt(len(b))
    cold = solver.solve(op, b + delta)
    assert np.array_equal(cold.free, first.free)  # support-stable step
    warm = solver.solve(op, b + delta, warm=(first.free, first.x))
    assert warm.converged
    assert warm.outer == 1
    assert warm.inner < cold.inner
    assert np.max(np.abs(warm.x - cold.x)) < 1e-6


def test_warm_start_survives_support_drift() -> None:
    """A warm start from a drifted support still reaches the right optimum."""
    a, b, _, _ = make_problem(80, 1e3, seed=1)
    first = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
    b2 = b + 0.3 * np.linalg.norm(b) * np.random.default_rng(0).standard_normal(len(b)) / np.sqrt(len(b))
    cold = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b2)
    warm = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b2, warm=(first.free, first.x))
    assert warm.converged
    assert np.max(np.abs(warm.x - cold.x)) < 1e-6


def test_rank_deficient_regularised_gram() -> None:
    """With m < n, any alpha > 0 restores well-posedness and recovery."""
    n, m, k = 80, 40, 20
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((m, n)) / np.sqrt(m)
    a0 = mat.T @ mat  # PSD, rank m < n
    perm = rng.permutation(n)
    x_star = np.zeros(n)
    x_star[perm[:k]] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
    for alpha in (0.05, 0.2):
        a = (1 - alpha) * a0 + alpha * np.eye(n)
        b = a @ x_star - s_star
        res = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
        assert res.converged
        assert np.max(np.abs(res.x - x_star)) < 1e-6


# --------------------------------------------------------------------------
# Operator interface: A enters only as a SymmetricOperator
# --------------------------------------------------------------------------


def test_gram_operator_recovers_planted_optimum() -> None:
    """The Gram-structured solve recovers a planted optimum matrix-free.

    Both the planting and the solve touch ``A = M^T M + ridge I`` only
    through products with ``M`` — the ``n x n`` Gram matrix never exists.
    """
    m, b, x_star = _plant_gram_problem(30, 60, ridge=1.0, seed=0)
    op = GramOperator(m, ridge=1.0)
    res = ActiveSetSolver(inner=CG()).solve(op, b)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert kkt_violation(op, b, res.x) < 1e-6


def test_exact_inner_uses_operator_solve_free() -> None:
    """``inner="exact"`` routes through ``op.solve_free`` for both backends."""
    a, b, x_star, _ = make_problem(60, 1e2, seed=1)
    res = ActiveSetSolver(inner=Exact()).solve(DenseOperator(a), b)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6

    m, b_g, x_star_g = _plant_gram_problem(30, 60, ridge=1.0, seed=2)
    res_g = ActiveSetSolver(inner=Exact()).solve(GramOperator(m, ridge=1.0), b_g)
    assert res_g.converged
    assert np.max(np.abs(res_g.x - x_star_g)) < 1e-6


def test_pcg_preconditions_from_operator_diag() -> None:
    """Jacobi PCG reads its preconditioner off ``op.diag`` — no matrix needed."""
    m, b, x_star = _plant_gram_problem(30, 60, ridge=1.0, seed=3)
    res = ActiveSetSolver(inner=Jacobi()).solve(GramOperator(m, ridge=1.0), b)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6


def test_pcg_without_diag_raises() -> None:
    """A backend without a cheap diagonal refuses Jacobi PCG."""
    a, b, _, _ = make_problem(30, 1e2, seed=0)
    with pytest.raises(NotImplementedError, match="diagonal"):
        ActiveSetSolver(inner=Jacobi()).solve(_NoDiagOperator(a), b)


def test_exact_inner_guards_singular_free_block() -> None:
    """``inner="exact"`` refuses a rank-deficient free block via ``rcond_free``.

    An un-ridged Gram operator with fewer rows than free variables has a
    singular free block on the initial all-free set; the guard turns the
    latent factorisation failure into a clean diagnostic.
    """
    rng = np.random.default_rng(0)
    m = rng.standard_normal((10, 30))  # rank 10 < n = 30
    b = m.T @ rng.standard_normal(10)
    with pytest.raises(ValueError, match="singular"):
        ActiveSetSolver(inner=Exact()).solve(GramOperator(m, ridge=0.0), b)


def test_exact_check_conditioning_false_skips_rcond_guard() -> None:
    """``Exact(check_conditioning=False)`` solves without ever calling ``rcond_free``.

    The guard is an ``O(|F|^3)`` eigendecomposition paid on every new free set;
    opting out (issue #32) drops it entirely while ``solve_free`` still returns
    the same optimum on a well-conditioned problem.
    """

    class _CountingOperator(DenseOperator):
        """Dense operator that tallies its ``rcond_free`` calls."""

        def __init__(self, a: np.ndarray) -> None:
            super().__init__(a)
            self.rcond_calls = 0

        def rcond_free(self, free: object) -> float:
            self.rcond_calls += 1
            return float(super().rcond_free(free))

    a, b, x_star, _ = make_problem(60, 1e2, seed=1)

    guarded = _CountingOperator(a)
    res_g = ActiveSetSolver(inner=Exact()).solve(guarded, b)
    assert guarded.rcond_calls > 0  # the default guard runs

    op = _CountingOperator(a)
    res = ActiveSetSolver(inner=Exact(check_conditioning=False)).solve(op, b)
    assert op.rcond_calls == 0  # opting out skips the eigendecomposition entirely
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert np.max(np.abs(res.x - res_g.x)) < 1e-9  # same solve, guard or not


def test_dimension_mismatch_is_rejected() -> None:
    """An operator whose dimension disagrees with len(b) is refused at entry."""
    a, b, _, _ = make_problem(20, 1e2, seed=0)
    op = DenseOperator(a)
    b_short = b[:-1]
    with pytest.raises(ValueError, match="dimension"):
        ActiveSetSolver(inner=CG()).solve(op, b_short)
    with pytest.raises(ValueError, match="dimension"):
        ActiveSetSolver(inner=CG()).solve_eq(op, b_short, np.ones((1, 19)), np.array([1.0]))
    with pytest.raises(ValueError, match="dimension"):
        kkt_violation(op, b_short, np.zeros(19))


def test_dense_arrays_are_rejected() -> None:
    """A dense array is refused with a pointer to DenseOperator."""
    a, b, _, _ = make_problem(20, 1e2, seed=0)
    with pytest.raises(TypeError, match="DenseOperator"):
        ActiveSetSolver(inner=CG()).solve(a, b)
    with pytest.raises(TypeError, match="DenseOperator"):
        ActiveSetSolver(inner=CG()).solve_eq(a, b, np.ones((1, 20)), np.array([1.0]))
    with pytest.raises(TypeError, match="DenseOperator"):
        kkt_violation(a, b, np.zeros(20))


# --------------------------------------------------------------------------
# Equality-augmented solve (B x = c via Schur complement)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 3, 8])
@pytest.mark.parametrize("kappa", [1e2, 1e4])
def test_recovers_planted_eq_optimum(p: int, kappa: float) -> None:
    """The Schur-complement loop recovers the planted optimum for general B."""
    a, b, b_eq, c_eq, x_star, _, _ = make_eq_problem(80, kappa, p, seed=p)
    res = ActiveSetSolver(inner=CG()).solve_eq(DenseOperator(a), b, b_eq, c_eq)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert np.linalg.norm(b_eq @ res.x - c_eq) < 1e-9  # feasible to machine precision
    assert res.lam is not None
    assert res.lam.shape == (p,)


def test_p_one_is_the_single_normalisation() -> None:
    """P = 1 with B = 1^T reproduces the bound-constrained solve on its own budget.

    With lambda* = 0 the bound-constrained optimum x* satisfies the KKT system
    of the equality-augmented problem whose budget is beta = 1^T x*, so the
    Schur-complement loop must return x* with a vanishing multiplier.
    """
    a, b, x_star, _ = make_problem(60, 1e3, seed=0)
    ones = np.ones((1, 60))
    beta = np.array([float(x_star.sum())])
    res = ActiveSetSolver(inner=CG()).solve_eq(DenseOperator(a), b, ones, beta)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-6
    assert res.lam is not None
    assert abs(float(res.lam[0])) < 1e-6
    assert abs(float(res.x.sum()) - float(beta[0])) < 1e-9


def test_eq_solver_accepts_gram_operator() -> None:
    """The equality-augmented loop runs matrix-free on a Gram operator."""
    m, b, x_star = _plant_gram_problem(30, 60, ridge=1.0, seed=4)
    ones = np.ones((1, 60))
    beta = np.array([float(x_star.sum())])
    res = ActiveSetSolver(inner=CG()).solve_eq(GramOperator(m, ridge=1.0), b, ones, beta)
    assert res.converged
    assert res.lam is not None
    assert abs(float(res.x.sum()) - float(beta[0])) < 1e-8
    assert np.max(np.abs(res.x - x_star)) < 1e-6


@pytest.mark.parametrize("inner", [CG(), Jacobi()])
def test_eq_warm_start_support_stable_single_step(inner: InnerSolver) -> None:
    """Across a support-stable step the warm eq loop takes one outer step and cuts inner iters.

    Both iterative inners consume the ``v0`` warm seed: the warm solve makes
    one outer step (the warm free set is already optimal) and its inner count
    falls below the cold solve's — the ``v0`` block converges from a near
    solution while the ``v1`` columns stay cold on both.
    """
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(80, 1e3, 3, seed=2)
    op = DenseOperator(a)
    solver = ActiveSetSolver(inner=inner)
    first = solver.solve_eq(op, b, b_eq, c_eq)
    delta = 1e-4 * np.linalg.norm(b) * np.ones_like(b) / np.sqrt(len(b))
    second_cold = solver.solve_eq(op, b + delta, b_eq, c_eq)
    assert np.array_equal(second_cold.free, first.free)  # support-stable step
    second_warm = solver.solve_eq(op, b + delta, b_eq, c_eq, warm=(first.free, first.x))
    assert second_warm.converged
    assert second_warm.outer == 1
    assert second_warm.inner < second_cold.inner
    assert np.max(np.abs(second_warm.x - second_cold.x)) < 1e-6
    assert np.linalg.norm(b_eq @ second_warm.x - c_eq) < 1e-9


def test_eq_warm_start_exact_single_outer_step() -> None:
    """``Exact`` has no inner seed, but the warm free set still gives one outer step.

    The direct solve has nothing to warm-start, so the reduction is purely the
    outer loop's: starting from the previous (already optimal) free set, the
    warm solve converges in a single outer step where the cold solve needs
    several — this pins the intentional "no inner seed for exact" behaviour.
    """
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(80, 1e3, 3, seed=2)
    op = DenseOperator(a)
    solver = ActiveSetSolver(inner=Exact())
    first = solver.solve_eq(op, b, b_eq, c_eq)
    delta = 1e-4 * np.linalg.norm(b) * np.ones_like(b) / np.sqrt(len(b))
    cold = solver.solve_eq(op, b + delta, b_eq, c_eq)
    assert np.array_equal(cold.free, first.free)  # support-stable step
    assert cold.outer > 1  # cold starts from the full free set and must shrink
    warm = solver.solve_eq(op, b + delta, b_eq, c_eq, warm=(first.free, first.x))
    assert warm.converged
    assert warm.outer == 1
    assert np.max(np.abs(warm.x - cold.x)) < 1e-6
    assert np.linalg.norm(b_eq @ warm.x - c_eq) < 1e-9


def test_eq_warm_start_survives_support_drift() -> None:
    """A warm start from a drifted support still reaches the right eq optimum."""
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(80, 1e3, 3, seed=3)
    op = DenseOperator(a)
    first = ActiveSetSolver(inner=CG()).solve_eq(op, b, b_eq, c_eq)
    b2 = b + 0.3 * np.linalg.norm(b) * np.random.default_rng(0).standard_normal(len(b)) / np.sqrt(len(b))
    cold = ActiveSetSolver(inner=CG()).solve_eq(op, b2, b_eq, c_eq)
    warm = ActiveSetSolver(inner=CG()).solve_eq(op, b2, b_eq, c_eq, warm=(first.free, first.x))
    assert warm.converged
    assert np.max(np.abs(warm.x - cold.x)) < 1e-6
    assert np.linalg.norm(b_eq @ warm.x - c_eq) < 1e-9


def test_eq_exact_inner_matches_cg() -> None:
    """The direct (:class:`Exact`) eq solve matches the CG solve.

    Both recover the planted optimum and settle on the same free set — the
    equality analogue of the inexactness lemma: the inner accuracy does not
    change the sign decisions of the outer loop.
    """
    a, b, b_eq, c_eq, x_star, _, _ = make_eq_problem(80, 1e4, 3, seed=5)
    op = DenseOperator(a)
    r_cg = ActiveSetSolver(inner=CG()).solve_eq(op, b, b_eq, c_eq)
    r_ex = ActiveSetSolver(inner=Exact()).solve_eq(op, b, b_eq, c_eq)
    assert r_ex.converged
    assert np.array_equal(r_ex.free, r_cg.free)
    assert np.max(np.abs(r_ex.x - x_star)) < 1e-6
    assert np.linalg.norm(b_eq @ r_ex.x - c_eq) < 1e-9


def test_eq_exact_guards_conditioning_once_per_free_set() -> None:
    """The exact eq path estimates ``rcond_free`` once per free set, not ``p + 1`` times.

    ``solve_eq`` drives the ``v0`` right-hand side plus one Schur-complement
    column per equality through the *same* free block each outer step. The
    conditioning guard depends only on the free set, so it must be paid once per
    free set (regression of #30 / #18); the ``p + 1`` ``solve_free`` calls prove
    the guard was genuinely shared, not that the column loop was skipped.
    """
    p = 8
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(80, 1e4, p, seed=4)

    class _CountingOperator(DenseOperator):  # type: ignore[misc]
        """Dense operator that tallies its ``rcond_free`` and ``solve_free`` calls."""

        def __init__(self, arr: np.ndarray) -> None:
            super().__init__(arr)
            self.rcond_calls = 0
            self.solve_calls = 0

        def rcond_free(self, free: object) -> float:
            """Count and delegate the conditioning estimate."""
            self.rcond_calls += 1
            return float(super().rcond_free(free))

        def solve_free(self, free: object, rhs: np.ndarray) -> np.ndarray:
            """Count and delegate the direct free-block solve."""
            self.solve_calls += 1
            return np.asarray(super().solve_free(free, rhs))

    op = _CountingOperator(a)
    res = ActiveSetSolver(inner=Exact()).solve_eq(op, b, b_eq, c_eq)
    assert res.converged
    assert op.solve_calls == res.outer * (p + 1)  # every column is still solved
    assert op.rcond_calls <= res.outer  # but the guard is shared across them
    assert op.rcond_calls < op.solve_calls  # the #30 waste (one guard per solve) is gone


def test_eq_pcg_inner_recovers_optimum() -> None:
    """The Jacobi-preconditioned eq solve recovers the same planted optimum."""
    a, b, b_eq, c_eq, x_star, _, _ = make_eq_problem(80, 1e4, 3, seed=6)
    r_pcg = ActiveSetSolver(inner=Jacobi()).solve_eq(DenseOperator(a), b, b_eq, c_eq)
    assert r_pcg.converged
    assert np.max(np.abs(r_pcg.x - x_star)) < 1e-6
    assert np.linalg.norm(b_eq @ r_pcg.x - c_eq) < 1e-9


def test_eq_nystrom_inner_recovers_optimum() -> None:
    """The Nyström-preconditioned eq solve recovers the same planted optimum.

    Each saddle step sketches ``A_F`` afresh for each of its ``p + 1`` right-hand
    sides (the sketch seed is fixed, so the preconditioner is identical each time).
    """
    a, b, b_eq, c_eq, x_star, _, _ = make_eq_problem(80, 1e4, 3, seed=6)
    inner = Nystrom(nystrom=NystromConfig(rank=15))
    r_ny = ActiveSetSolver(inner=inner).solve_eq(DenseOperator(a), b, b_eq, c_eq)
    assert r_ny.converged
    assert np.max(np.abs(r_ny.x - x_star)) < 1e-6
    assert np.linalg.norm(b_eq @ r_ny.x - c_eq) < 1e-9


def test_eq_pcg_beats_cg_under_diagonal_scaling() -> None:
    """On a diagonally ill-scaled eq problem PCG needs fewer inner iterations."""
    a, b, b_eq, c_eq, x_star, _, _ = make_scaled_eq_problem(80, 1e2, 1e6, 3, seed=7)
    op = DenseOperator(a)
    r_cg = ActiveSetSolver(inner=CG()).solve_eq(op, b, b_eq, c_eq)
    r_pcg = ActiveSetSolver(inner=Jacobi()).solve_eq(op, b, b_eq, c_eq)
    assert r_pcg.converged
    assert np.max(np.abs(r_pcg.x - x_star)) < 1e-6
    assert np.max(np.abs(r_cg.x - x_star)) < 1e-6
    # A comfortable margin, not a bare `<`: the 1e6 diagonal spread makes the
    # Jacobi win large enough to survive BLAS-dependent iteration-count drift.
    assert r_pcg.inner <= 0.7 * r_cg.inner


def test_eq_track_records_trajectory() -> None:
    """``track=True`` records the visited free sets, ending on the converged one."""
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(60, 1e3, 3, seed=8)
    res = ActiveSetSolver(inner=CG(), config=ActiveSetConfig(track=True)).solve_eq(DenseOperator(a), b, b_eq, c_eq)
    assert res.converged
    assert res.traj is not None
    assert len(res.traj) == res.outer  # one recorded free set per outer step
    assert np.array_equal(res.traj[-1], np.flatnonzero(res.free))  # last = converged support


def test_eq_max_outer_caps_the_loop() -> None:
    """``max_outer`` stops the loop early and reports non-convergence."""
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(80, 1e5, 3, seed=9)
    res = ActiveSetSolver(inner=CG(), config=ActiveSetConfig(max_outer=1)).solve_eq(DenseOperator(a), b, b_eq, c_eq)
    assert not res.converged
    assert res.outer == 1


def test_eq_reduced_gradient_certifies() -> None:
    """At the exit, x >= 0 and s = Ax - b - B^T lam >= 0 hold to tolerance.

    The certificate tolerance is 1e-6, not the solver's tol=1e-8: the dual
    test only guards bound indices, while on free indices s carries the inner
    CG residual, which varies with the NumPy/BLAS version.
    """
    a, b, b_eq, c_eq, _, _, _ = make_eq_problem(60, 1e3, 3, seed=1)
    res = ActiveSetSolver(inner=CG()).solve_eq(DenseOperator(a), b, b_eq, c_eq)
    assert res.lam is not None
    s = a @ res.x - b - b_eq.T @ res.lam
    assert float(np.min(res.x)) > -1e-6
    assert float(np.min(s)) > -1e-6
    assert float(np.max(np.abs(res.x * s))) < 1e-6


# --------------------------------------------------------------------------
# Anti-cycling guard / Bland fallback (finite-termination guarantee)
# --------------------------------------------------------------------------

_ADVERSARIAL_N = 20
_ADVERSARIAL_SEEDS = range(30)


def test_pure_block_pivoting_cycles_somewhere() -> None:
    """With the guard disabled, at least one seed revisits a free set and stalls."""
    cycled = 0
    for seed in _ADVERSARIAL_SEEDS:
        a, b = make_adversarial(_ADVERSARIAL_N, seed=seed)
        config = ActiveSetConfig(p_max=10**9, max_outer=300, track=True)
        res = ActiveSetSolver(config=config, inner=Exact()).solve(DenseOperator(a), b)
        assert res.traj is not None
        if not res.converged and len(res.traj) != len(set(res.traj)):
            cycled += 1
    assert cycled > 0


def test_guarded_loop_terminates_everywhere() -> None:
    """The guarded loop (default patience) terminates and certifies on all seeds."""
    fired = 0
    for seed in _ADVERSARIAL_SEEDS:
        a, b = make_adversarial(_ADVERSARIAL_N, seed=seed)
        res = ActiveSetSolver(inner=CG()).solve(DenseOperator(a), b)
        assert res.converged
        assert kkt_violation(DenseOperator(a), b, res.x) < 1e-6
        fired += res.fallback > 0
    assert fired > 0  # the fallback is genuinely exercised, not dormant


def test_guarded_eq_loop_terminates_everywhere() -> None:
    """The equality-augmented loop terminates and certifies on all seeds.

    The anti-correlated family under the single normalisation ``1^T x = 1``
    drives the batch path of :meth:`ActiveSetSolver.solve_eq` through patience exhaustion
    into the Bland fallback on part of the seeds; the guarded loop must
    terminate at a KKT-certified point on all of them.
    """
    b_eq = np.ones((1, _ADVERSARIAL_N))
    c_eq = np.array([1.0])
    fired = 0
    for seed in _ADVERSARIAL_SEEDS:
        a, b = make_adversarial(_ADVERSARIAL_N, seed=seed)
        res = ActiveSetSolver(inner=CG()).solve_eq(DenseOperator(a), b, b_eq, c_eq)
        assert res.converged
        assert res.lam is not None
        s = a @ res.x - b - b_eq.T @ res.lam
        assert float(np.min(res.x)) > -1e-6
        assert float(np.min(s)) > -1e-6
        assert abs(float(res.x.sum()) - 1.0) < 1e-6
        fired += res.fallback > 0
    assert fired > 0  # the eq-variant fallback is genuinely exercised, not dormant


# --------------------------------------------------------------------------
# Planted-optimum generators (tests/problems.py)
# --------------------------------------------------------------------------


def test_make_problem_plants_kkt_point() -> None:
    """The planted pair satisfies the KKT system exactly."""
    a, b, x_star, s_star = make_problem(50, 1e3, seed=0)
    assert np.allclose(a @ x_star - b, s_star)
    assert float(np.min(x_star)) >= 0.0
    assert float(np.min(s_star)) >= 0.0
    assert float(np.max(np.abs(x_star * s_star))) == 0.0
    assert kkt_violation(DenseOperator(a), b, x_star) < 1e-12


def test_make_problem_condition_number() -> None:
    """The spectrum spans [1, kappa] as prescribed."""
    kappa = 1e4
    a, _, _, _ = make_problem(50, kappa, seed=1)
    eig = np.linalg.eigvalsh(a)
    assert eig[0] > 0
    assert np.isclose(eig[-1] / eig[0], kappa, rtol=1e-6)


def test_make_eq_problem_plants_kkt_point() -> None:
    """The planted equality-augmented triple satisfies its KKT system."""
    a, b, b_eq, c_eq, x_star, lam_star, s_star = make_eq_problem(50, 1e3, 3, seed=0)
    assert np.allclose(a @ x_star - b_eq.T @ lam_star - b, s_star)
    assert np.allclose(b_eq @ x_star, c_eq)
    assert float(np.max(np.abs(x_star * s_star))) == 0.0


def test_make_adversarial_is_spd() -> None:
    """The adversarial design is strictly positive definite (a P-matrix)."""
    a, b = make_adversarial(20, seed=0)
    assert np.linalg.eigvalsh(a)[0] > 0
    assert b.shape == (20,)


def test_make_scaled_problem_spread() -> None:
    """The scaled operator's conditioning grows with the diagonal spread."""
    a1, _, _ = make_scaled_problem(40, 10.0, 1.0, seed=0)
    a2, _, _ = make_scaled_problem(40, 10.0, 1e4, seed=0)
    k1 = float(np.linalg.cond(a1))
    k2 = float(np.linalg.cond(a2))
    assert k2 > 10 * k1
