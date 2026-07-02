"""Tests of the anti-cycling guard on the adversarial anti-correlated family.

Pure block principal pivoting (patience unbounded) provably cycles on part of
this family — a previously seen free set recurs — while the guarded loop with
default patience terminates on every seed. This is the paper's Panel H as a
regression test: the fallback path is load-bearing and must stay exercised.
"""

from nncg import kkt_violation, make_adversarial, solve_nnqp

N = 20
SEEDS = range(30)


def test_pure_block_pivoting_cycles_somewhere() -> None:
    """With the guard disabled, at least one seed revisits a free set and stalls."""
    cycled = 0
    for seed in SEEDS:
        a, b = make_adversarial(N, seed=seed)
        res = solve_nnqp(a, b, p_max=10**9, inner="exact", max_outer=300, track=True)
        assert res.traj is not None
        if not res.converged and len(res.traj) != len(set(res.traj)):
            cycled += 1
    assert cycled > 0


def test_guarded_loop_terminates_everywhere() -> None:
    """The guarded loop (default patience) terminates and certifies on all seeds."""
    fired = 0
    for seed in SEEDS:
        a, b = make_adversarial(N, seed=seed)
        res = solve_nnqp(a, b)
        assert res.converged
        assert kkt_violation(a, b, res.x) < 1e-6
        fired += res.fallback > 0
    assert fired > 0  # the fallback is genuinely exercised, not dormant
