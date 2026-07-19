# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.13",
#     "numpy>=2.0.0",
#     "matplotlib",
#     "cvx-linalg>=0.9.6",
#     "nncg",
# ]
#
# [tool.uv.sources]
# nncg = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Comparing inner solvers: CG, Jacobi, Nyström, GlobalNystrom

    Every outer active-set step needs one free-block solve
    $A_{FF}\, x_F = b_F$. `nncg` never fixes *how* that solve happens — it is
    delegated to a pluggable `InnerSolver`. This notebook compares the four
    built-in Krylov-based choices head to head, on a problem shaped to show
    exactly what each buys you:

    | solver | preconditioner | cost pattern |
    |---|---|---|
    | `CG` | none (identity) | pays the full condition number of $A_{FF}$ |
    | `Jacobi` | $r \mapsto \mathrm{diag}(A)^{-1}_F\, r$ | strips a *bad diagonal scaling* |
    | `Nystrom` | randomized rank-$r$ sketch of $A_{FF}$ | deflates a few eigenvalues, resketched every step |
    | `GlobalNystrom` | same sketch, of the full $A$, masked to $F$ | identical deflation, sketched once ever |

    `Nystrom` and `GlobalNystrom` target the same spectral structure — a
    handful of eigenvalues that dwarf the rest — but differ in *when* they pay
    for the sketch. `Nystrom` resketches $A[F,F]$ from scratch on every
    outer step; `GlobalNystrom` sketches the whole operator once (restricting
    a rank-$r$ factorisation to a principal submatrix is exact, so masking
    rows of one global basis is a valid free-block preconditioner with no
    further matrix-free products against $A$). That should be a wash on a
    single solve and a clear win across many.
    """)
    return


@app.cell
def _():
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from cvx.linalg import DenseOperator

    from nncg import (
        CG,
        ActiveSetConfig,
        ActiveSetSolver,
        GlobalNystrom,
        Jacobi,
        Nystrom,
        NystromConfig,
    )

    return (
        ActiveSetConfig,
        ActiveSetSolver,
        CG,
        DenseOperator,
        GlobalNystrom,
        Jacobi,
        Nystrom,
        NystromConfig,
        np,
        plt,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## The test problem: a few dominant eigenvalues over a spread tail

    Neither Nyström variant helps on a smoothly decaying spectrum — there is
    nothing cheap to deflate. The shape that showcases them is a handful of
    eigenvalues sitting far above a spread-out tail: capturing just those few
    directions collapses the condition number `Nystrom`/`GlobalNystrom`
    actually run at down to the *tail's* spread, while plain `CG` still pays
    for the full range top-to-bottom. A planted optimum lets us check every
    solver recovers the same, correct $x^\star$.
    """)
    return


@app.cell
def _(np):
    def make_problem_with_spectrum(n, n_dominant, top_gap, tail_kappa, support_frac=0.5, seed=0):
        """Planted-optimum SPD problem with a spectral gap.

        `n_dominant` eigenvalues sit at `top_gap * tail_kappa` and above, over a
        geometric tail spanning `[1, tail_kappa]`.
        """
        rng = np.random.default_rng(seed)
        top = tail_kappa * np.geomspace(top_gap, 2.0, n_dominant)
        tail = np.geomspace(1.0, tail_kappa, n - n_dominant)
        eig = np.concatenate([top, tail])
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        a = (q * eig) @ q.T
        a = 0.5 * (a + a.T)

        k = max(1, round(support_frac * n))
        perm = rng.permutation(n)
        supp = perm[:k]
        x_star = np.zeros(n)
        x_star[supp] = rng.uniform(0.5, 1.5, size=k)
        s_star = np.zeros(n)
        s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
        b = a @ x_star - s_star
        return a, b, x_star

    return (make_problem_with_spectrum,)


@app.cell
def _(mo):
    n_slider = mo.ui.slider(50, 800, value=300, step=10, label="dimension $n$")
    n_dom_slider = mo.ui.slider(1, 8, value=3, step=1, label="# dominant eigenvalues")
    top_gap_slider = mo.ui.slider(10, 1000, value=100, step=10, label="dominant / tail gap")
    tail_kappa_slider = mo.ui.slider(1, 4, value=2, step=1, label=r"tail spread $\log_{10}\kappa$")
    rank_slider = mo.ui.slider(1, 10, value=4, step=1, label="sketch rank")
    seed_slider = mo.ui.slider(0, 30, value=0, step=1, label="seed")
    mo.vstack([n_slider, n_dom_slider, top_gap_slider, tail_kappa_slider, rank_slider, seed_slider])
    return (
        n_dom_slider,
        n_slider,
        rank_slider,
        seed_slider,
        tail_kappa_slider,
        top_gap_slider,
    )


@app.cell
def _(
    make_problem_with_spectrum,
    n_dom_slider,
    n_slider,
    seed_slider,
    tail_kappa_slider,
    top_gap_slider,
):
    a, b, x_star = make_problem_with_spectrum(
        n=n_slider.value,
        n_dominant=n_dom_slider.value,
        top_gap=top_gap_slider.value,
        tail_kappa=10.0**tail_kappa_slider.value,
        seed=seed_slider.value,
    )
    return a, b, x_star


@app.cell
def _(
    ActiveSetConfig,
    ActiveSetSolver,
    CG,
    DenseOperator,
    GlobalNystrom,
    Jacobi,
    Nystrom,
    NystromConfig,
    a,
    b,
    np,
    rank_slider,
    time,
    x_star,
):
    _op = DenseOperator(a)
    _cfg = ActiveSetConfig(tol=1e-8)
    _rank = rank_slider.value

    solvers = {
        "CG (no preconditioner)": CG(),
        "Jacobi": Jacobi(),
        "Nystrom (resketched per step)": Nystrom(nystrom=NystromConfig(rank=_rank, seed=0)),
        "GlobalNystrom (sketched once)": GlobalNystrom(nystrom=NystromConfig(rank=_rank, seed=0)),
    }

    results = {}
    for _name, _inner in solvers.items():
        _t0 = time.perf_counter()
        _res = ActiveSetSolver(inner=_inner, config=_cfg).solve(_op, b)
        _dt = time.perf_counter() - _t0
        results[_name] = {
            "outer": _res.outer,
            "inner": _res.inner,
            "time_ms": _dt * 1e3,
            "error": float(np.max(np.abs(_res.x - x_star))),
            "converged": _res.converged,
        }
    return (results,)


@app.cell
def _(mo, results):
    _rows = "\n".join(
        f"| {name} | {r['outer']} | {r['inner']} | {r['time_ms']:.2f} | {r['error']:.2e} | {r['converged']} |"
        for name, r in results.items()
    )
    mo.md(
        f"""
        | solver | outer steps | total inner CG iters | wall time (ms) | max error vs $x^\\star$ | converged |
        |---|---|---|---|---|---|
        {_rows}

        All four reach the same planted optimum (the outer trajectory only depends
        on the violator tests, not on how accurately the inner solve runs) —
        what differs is how many inner CG iterations it took to get there, and how
        long that took in wall time.
        """
    )
    return


@app.cell
def _(plt, results):
    _names = list(results.keys())
    _inner_counts = [results[n]["inner"] for n in _names]

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _bars = _ax.barh(_names, _inner_counts, color=["#888", "#5b8", "#58b", "#b58"])
    _ax.set_xlabel("total inner CG iterations (lower is better)")
    _ax.set_title("Inner iteration count by preconditioner")
    _ax.bar_label(_bars, fmt="%d", padding=3)
    _ax.invert_yaxis()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where `GlobalNystrom` actually wins: reusing one operator

    A single solve is close to a wash: `GlobalNystrom`'s one-time sketch of
    the *full* operator is more expensive than `Nystrom`'s first sketch of a
    (usually smaller) free block, even though it is never paid again. The
    difference shows up once the *same* operator is solved repeatedly — a
    parameter sweep, successive warm starts, a sensitivity analysis. `Nystrom`
    pays its sketch cost on every outer step of every solve; `GlobalNystrom`
    pays it once, ever, and every later solve only pays the cheap
    $O(|F|\,\mathrm{rank})$ masking cost.

    The slider below repeats the *same* operator `A` across many independent
    right-hand sides, sharing one solver instance (and hence, for
    `GlobalNystrom`, one cached sketch) across all of them.
    """)
    return


@app.cell
def _(mo):
    n_solves_slider = mo.ui.slider(2, 40, value=15, step=1, label="number of right-hand sides")
    n_solves_slider
    return (n_solves_slider,)


@app.cell
def _(
    ActiveSetConfig,
    ActiveSetSolver,
    GlobalNystrom,
    Nystrom,
    NystromConfig,
    a,
    n_slider,
    n_solves_slider,
    np,
    rank_slider,
    time,
):
    from cvx.linalg import DenseOperator as _DenseOperator

    _op = _DenseOperator(a)
    _cfg = ActiveSetConfig(tol=1e-8)
    _rank = rank_slider.value
    _rng = np.random.default_rng(1)
    _rhs = [_rng.standard_normal(n_slider.value) for _ in range(n_solves_slider.value)]

    reuse_results = {}
    for _label, _inner in (
        ("Nystrom (resketched every solve)", Nystrom(nystrom=NystromConfig(rank=_rank, seed=0))),
        ("GlobalNystrom (sketched once, shared)", GlobalNystrom(nystrom=NystromConfig(rank=_rank, seed=0))),
    ):
        _cumulative = []
        _t0 = time.perf_counter()
        for _b in _rhs:
            ActiveSetSolver(inner=_inner, config=_cfg).solve(_op, _b)
            _cumulative.append((time.perf_counter() - _t0) * 1e3)
        reuse_results[_label] = _cumulative
    return (reuse_results,)


@app.cell
def _(plt, reuse_results):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _label, _cumulative in reuse_results.items():
        _ax.plot(range(1, len(_cumulative) + 1), _cumulative, marker="o", label=_label)
    _ax.set_xlabel("right-hand side #")
    _ax.set_ylabel("cumulative wall time (ms)")
    _ax.set_title("Cumulative cost of solving the same operator repeatedly")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    `GlobalNystrom`'s line should start with a one-time bump (the full-operator
    sketch, paid on the first right-hand side) and then grow in step with
    `Nystrom`'s per-block resketching cost — flattening the gap between them as
    more right-hand sides are solved. On a single, one-off solve there is
    little reason to prefer `GlobalNystrom` over `Nystrom`; the moment the
    *same* operator is reused, the sketch-once design starts paying for
    itself.

    ### Where to go next

    - **[Active-set methods](01_active_set_methods.html)** — the outer loop
      both solvers plug into.
    - **[Equality constraints](02_equality_constraints.html)** — the
      Schur-complement path, where every outer step needs $p+1$ free-block
      solves and the sketch-once saving compounds further.
    - **[Test problems](03_test_problems.html)** — the planted-optimum
      generators this notebook's problem builder is modelled on.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
