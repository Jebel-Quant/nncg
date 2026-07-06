"""Tests of the anti-cycling guard on the adversarial anti-correlated family.

Pure block principal pivoting (patience unbounded) provably cycles on part of
this family — a previously seen free set recurs — while the guarded loop with
default patience terminates on every seed. This is the paper's Panel H as a
regression test: the fallback path is load-bearing and must stay exercised.
"""

import numpy as np
from cvx.linalg import DenseOperator

from nncg import CG, ActiveSetConfig, ActiveSetSolver, Exact, kkt_violation
from tests.problems import make_adversarial

N = 20
SEEDS = range(30)


def test_pure_block_pivoting_cycles_somewhere() -> None:
    """With the guard disabled, at least one seed revisits a free set and stalls."""
    cycled = 0
    for seed in SEEDS:
        a, b = make_adversarial(N, seed=seed)
        config = ActiveSetConfig(p_max=10**9, max_outer=300, track=True)
        res = ActiveSetSolver(config=config, inner=Exact()).solve(DenseOperator(a), b)
        assert res.traj is not None
        if not res.converged and len(res.traj) != len(set(res.traj)):
            cycled += 1
    assert cycled > 0


def test_guarded_loop_terminates_everywhere() -> None:
    """The guarded loop (default patience) terminates and certifies on all seeds."""
    fired = 0
    for seed in SEEDS:
        a, b = make_adversarial(N, seed=seed)
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
    b_eq = np.ones((1, N))
    c_eq = np.array([1.0])
    fired = 0
    for seed in SEEDS:
        a, b = make_adversarial(N, seed=seed)
        res = ActiveSetSolver(inner=CG()).solve_eq(DenseOperator(a), b, b_eq, c_eq)
        assert res.converged
        assert res.lam is not None
        s = a @ res.x - b - b_eq.T @ res.lam
        assert float(np.min(res.x)) > -1e-6
        assert float(np.min(s)) > -1e-6
        assert abs(float(res.x.sum()) - 1.0) < 1e-6
        fired += res.fallback > 0
    assert fired > 0  # the eq-variant fallback is genuinely exercised, not dormant
