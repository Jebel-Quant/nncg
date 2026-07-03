# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=2.0.0",
#     "matplotlib",
#     "cvx-linalg>=0.9.6",
#     "nncg",
# ]
#
# [tool.uv.sources]
# nncg = { path = "../../..", editable = true }
# ///
#
# The `tests.problems` module lives at the repository root (outside the
# installed package); the import cell below locates the repo root and puts it
# on sys.path, so this notebook runs both under `make book` (marimo export
# --sandbox, cwd = book/marimo/notebooks) and via `uv run marimo edit`.
import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # The planted-optimum test problems

        This notebook documents `tests/problems.py` — the synthetic problem
        generators that power `nncg`'s test suite *and* the numerical study of the
        accompanying paper. There are four, each engineered to stress a different
        part of the solver:

        | generator | plants | stresses |
        |---|---|---|
        | `make_problem` | $(x^\star, s^\star)$ | correctness across condition numbers $\kappa$ |
        | `make_eq_problem` | $(x^\star, \lambda^\star, s^\star)$ | the equality-augmented / Schur path |
        | `make_adversarial` | *nothing* | the Bland fallback (BPP cycles here) |
        | `make_scaled_problem` | $(x^\star)$ | Jacobi preconditioning (PCG vs CG) |

        The unifying idea is **planting a known optimum**. Rather than solve a
        problem and hope the answer is right, we *construct* the answer first and
        derive a problem that has it — then a solver is correct iff it recovers the
        plant. It is the honest way to test a bound-constrained QP solver.
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from cvx.linalg import DenseOperator

    from nncg import kkt_violation, solve_nnqp, solve_nnqp_eq

    return DenseOperator, kkt_violation, np, plt, solve_nnqp, solve_nnqp_eq


@app.cell
def _():
    # Import the real generators from the test suite. Requires the repo root on
    # sys.path; add it defensively so the notebook runs from anywhere.
    import sys
    from pathlib import Path

    _here = Path.cwd()
    for _cand in (_here, *_here.parents):
        if (_cand / "tests" / "problems.py").exists():
            if str(_cand) not in sys.path:
                sys.path.insert(0, str(_cand))
            break

    from tests.problems import (
        make_adversarial,
        make_eq_problem,
        make_problem,
        make_scaled_problem,
    )

    return make_adversarial, make_eq_problem, make_problem, make_scaled_problem


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The planting recipe (why it works)

        The KKT system of $\min_{x\ge0}\tfrac12 x^\top A x - b^\top x$ is the LCP

        $$
        x \ge 0,\quad s = Ax - b \ge 0,\quad x \odot s = 0 .
        $$

        To *plant* a solution, choose a complementary pair $(x^\star, s^\star)$
        directly: pick a support $S$, set $x^\star > 0$ on $S$ and $0$ off it, and
        $s^\star = 0$ on $S$ and $>0$ off it — so $x^\star \odot s^\star = 0$ by
        construction. Then define the linear term by **reverse-engineering** the
        stationarity relation $s = Ax - b$:

        $$
        b \;=\; A x^\star - s^\star .
        $$

        Now $(x^\star, s^\star)$ satisfies the LCP exactly, and since $A \succ 0$
        makes the minimiser unique, $x^\star$ *is* the answer. Every generator
        below is a variation on this one move.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. `make_problem` — the workhorse

        $A = Q \operatorname{diag}(\mathrm{eig})\, Q^\top$ with a Haar-random
        orthogonal $Q$ and a **geometric spectrum** on $[1, \kappa]$, so the
        condition number is exactly the knob `kappa`. A support of size
        $\operatorname{round}(\texttt{support\_frac}\cdot n)$ is drawn; $x^\star$ is
        uniform in $[0.5, 1.5]$ there and $0$ elsewhere, $s^\star$ the mirror, and
        $b = A x^\star - s^\star$.

        Returns `(a, b, x_star, s_star)`. It drives the correctness sweeps — a
        solver must recover `x_star` at every $\kappa$.
        """
    )
    return


@app.cell
def _(DenseOperator, kkt_violation, make_problem, np, solve_nnqp):
    a1, b1, x1, s1 = make_problem(n=60, kappa=1e4, support_frac=0.5, seed=0)
    op1 = DenseOperator(a1)
    r1 = solve_nnqp(op1, b1)

    print("make_problem(n=60, kappa=1e4, support_frac=0.5)")
    print("  planted support size :", int((x1 > 0).sum()))
    print("  recovered support    :", int((r1.x > 1e-9).sum()))
    print("  || x - x_star ||     :", float(np.linalg.norm(r1.x - x1)))
    print("  complementarity plant:", float(np.max(np.abs(x1 * s1))), "(should be 0)")
    print("  KKT violation        :", kkt_violation(op1, b1, r1.x))
    print("  outer / inner        :", r1.outer, "/", r1.inner)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Sweep the condition number and watch the solver stay exact while the inner
        CG iterations climb at the Krylov rate $O(\sqrt{\kappa})$.
        """
    )
    return


@app.cell
def _(DenseOperator, kkt_violation, make_problem, np, plt, solve_nnqp):
    kappas = np.geomspace(1e1, 1e6, 12)
    errs, inner_iters = [], []
    for _kap in kappas:
        _a, _b, _x, _s = make_problem(n=80, kappa=float(_kap), seed=1)
        _op = DenseOperator(_a)
        _r = solve_nnqp(_op, _b)
        errs.append(kkt_violation(_op, _b, _r.x))
        inner_iters.append(_r.inner)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    _a1.loglog(kappas, np.maximum(errs, 1e-16), "o-", color="#11D48E")
    _a1.set_xlabel(r"condition number $\kappa$")
    _a1.set_ylabel("KKT violation")
    _a1.set_title("Correct at every conditioning")
    _a2.semilogx(kappas, inner_iters, "o-", color="#0b7d55")
    _a2.set_xlabel(r"condition number $\kappa$")
    _a2.set_ylabel("total inner CG iterations")
    _a2.set_title(r"CG cost $\sim O(\sqrt{\kappa})$")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. `make_eq_problem` — the equality-augmented plant

        The KKT system now carries a multiplier: with $B \in \mathbb{R}^{p\times
        n}$,

        $$
        x \ge 0,\quad s = Ax - b - B^\top\lambda \ge 0,\quad x\odot s = 0,\quad Bx = c .
        $$

        The generator plants $(x^\star, \lambda^\star, s^\star)$: a support of size
        **at least $p+1$** (so $B_F$ is generically full row rank — a prerequisite
        for the Schur complement to be SPD), $s^\star = 0$ there and positive off
        it, an arbitrary $\lambda^\star$, and then

        $$
        b = A x^\star - B^\top\lambda^\star - s^\star, \qquad c = B x^\star .
        $$

        Returns `(a, b, b_eq, c_eq, x_star, lam_star, s_star)`. Note the solver
        recovers `x_star` exactly, but `lam` need not equal `lam_star` unless the
        planted support is the *unique* optimal one — the multiplier is pinned by
        the active support, which is what actually matters.
        """
    )
    return


@app.cell
def _(DenseOperator, make_eq_problem, np, solve_nnqp_eq):
    a_e, b_e, b_eq_e, c_e, x_e, lam_e, _s_e = make_eq_problem(n=60, kappa=1e3, p=3, seed=2)
    op_e = DenseOperator(a_e)
    r_e = solve_nnqp_eq(op_e, b_e, b_eq_e, c_e)

    print("make_eq_problem(n=60, kappa=1e3, p=3)")
    print("  || x - x_star ||   :", float(np.linalg.norm(r_e.x - x_e)))
    print("  || B x - c ||      :", float(np.linalg.norm(b_eq_e @ r_e.x - c_e)))
    print("  planted lam_star   :", np.round(lam_e, 4))
    print("  recovered lam      :", np.round(r_e.lam, 4))
    print("  support >= p+1     :", int((x_e > 0).sum()), ">=", 3 + 1)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. `make_adversarial` — forcing the Bland fallback

        This is the one generator that plants **no** optimum; it exists to break
        the fast path. Columns arrive in near-**anti-parallel** pairs,

        $$
        M = [\,M_0 \mid -M_0 + \texttt{noise}\cdot E\,], \qquad A = M^\top M + \texttt{ridge}\cdot I,
        $$

        so pushing one variable to its bound flips the sign of its partner. A pure
        block-principal-pivoting exchange systematically over-shoots and **cycles**
        — it revisits a working set it has already seen and loops forever. The ridge
        keeps $A$ a strictly positive-definite P-matrix.

        The guarded loop in `nncg` terminates on all of these: when a batch step
        stops making progress and patience runs out, it takes a single least-index
        Bland pivot, which cannot cycle. Returns `(a, b)` only — certify the solve
        with `kkt_violation`.
        """
    )
    return


@app.cell
def _(DenseOperator, kkt_violation, make_adversarial, solve_nnqp):
    # Scan seeds; count how many need the fallback, and confirm ALL converge.
    n_seeds = 40
    fired, all_ok, worst_kkt = 0, True, 0.0
    example_fallback = None
    for _seed in range(n_seeds):
        _a, _b = make_adversarial(n=40, seed=_seed)
        _op = DenseOperator(_a)
        _r = solve_nnqp(_op, _b)
        _k = kkt_violation(_op, _b, _r.x)
        worst_kkt = max(worst_kkt, _k)
        all_ok = all_ok and _r.converged and _k < 1e-6
        if _r.fallback > 0:
            fired += 1
            if example_fallback is None:
                example_fallback = (_seed, _r.fallback, _r.outer)

    print(f"scanned {n_seeds} adversarial seeds (n=40)")
    print(f"  seeds that triggered the Bland fallback : {fired}/{n_seeds}")
    print(f"  all converged to KKT < 1e-6            : {all_ok}")
    print(f"  worst KKT violation                    : {worst_kkt:.2e}")
    if example_fallback:
        _sd, _fb, _ot = example_fallback
        print(f"  e.g. seed {_sd}: {_fb} fallback pivots in {_ot} outer steps")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        On a sizeable fraction of seeds the fallback fires — and on **every** seed
        the solve still terminates at a certified optimum. That is the termination
        guarantee earning its keep; `tests/test_fallback.py` locks this behaviour
        in as a regression, because the fallback path is the load-bearing component
        of the finite-termination proof.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. `make_scaled_problem` — a case for preconditioning

        $A = D^{1/2}(Q\Lambda Q^\top) D^{1/2}$: a **well-conditioned core** $Q\Lambda
        Q^\top$ (condition number `kappa_core`) wrapped in a **bad diagonal
        scaling** $D$ whose entries span $[1, \texttt{spread}]$. The full matrix is
        badly conditioned — roughly `kappa_core * spread` — so plain CG suffers. But
        the ill-conditioning is *entirely* the diagonal, so **Jacobi
        preconditioning** ($M^{-1}=\operatorname{diag}(A)^{-1}$) removes it and PCG
        runs at the *core's* condition number regardless of the spread.

        `solve_nnqp` selects the inner solver with `inner="cg"` (default) or
        `inner="pcg"`. The sweep below contrasts their inner iteration counts as the
        diagonal spread grows.
        """
    )
    return


@app.cell
def _(DenseOperator, make_scaled_problem, np, plt, solve_nnqp):
    spreads = np.geomspace(1e0, 1e6, 10)
    cg_it, pcg_it = [], []
    for _sp in spreads:
        _a, _b, _x = make_scaled_problem(n=80, kappa_core=1e2, spread=float(_sp), seed=4)
        _op = DenseOperator(_a)
        cg_it.append(solve_nnqp(_op, _b, inner="cg").inner)
        pcg_it.append(solve_nnqp(_op, _b, inner="pcg").inner)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.loglog(spreads, cg_it, "o-", color="#0b7d55", label='inner="cg"')
    _ax.loglog(spreads, pcg_it, "s-", color="#11D48E", label='inner="pcg" (Jacobi)')
    _ax.set_xlabel(r"diagonal spread of $D$")
    _ax.set_ylabel("total inner iterations")
    _ax.set_title("Jacobi PCG is immune to diagonal scaling")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Plain CG's iteration count climbs with the spread; Jacobi-PCG stays flat at
        the core's cost. This is the numerical evidence behind the `inner="pcg"`
        option and the `GramOperator`/`diag` machinery.

        ### Recap

        These four generators are deliberately kept **outside** the installed
        package (in `tests/problems.py`, next to the tests) yet importable — so they
        serve both the CI test suite and exploratory notebooks like this one. Every
        mathematical claim in the paper that `nncg` implements has a test built on
        one of them:

        - `make_problem` → correctness across $\kappa$,
        - `make_eq_problem` → the equality-augmented Schur path for $p\in\{1,3,8\}$,
        - `make_adversarial` → the provably necessary Bland fallback,
        - `make_scaled_problem` → the Jacobi-preconditioning win.
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
