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

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Active-set vs MPRGP

        `nncg` ships **two** solvers for the same strictly convex non-negative
        quadratic program

        $$
        \min_{x \ge 0}\ \tfrac12\, x^\top A x - b^\top x,
        \qquad A = A^\top \succ 0 .
        $$

        - **`solve_nnqp`** — the primal-dual **active-set** loop (block principal
          pivoting with a Bland fallback). It *guesses the optimal free set*, solves
          an unconstrained reduced system $A_{FF}x_F=b_F$ on it by CG, and corrects
          the guess. See the [active-set notebook](01_active_set_methods.html).
        - **`solve_nnqp_mprgp`** — **MPRGP** (Dostál & Schöberl), a matrix-free
          *projection* method that interleaves conjugate-gradient, **expansion** and
          **proportioning** steps under a proportioning test, never solving a face
          exactly.

        They target the same minimiser with very different machinery. This notebook
        puts them head to head: do they agree, how do they *scale*, and **when is
        each faster**. The short version — measured below — is that on these dense
        SPD problems the active-set loop is the stronger default, and MPRGP is the
        complementary tool for the regimes active-set methods find hard.
        """
    )
    return


@app.cell
def _():
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from cvx.linalg import DenseOperator, power_iteration

    from nncg import kkt_violation, solve_nnqp, solve_nnqp_mprgp

    return (
        DenseOperator,
        kkt_violation,
        np,
        plt,
        power_iteration,
        solve_nnqp,
        solve_nnqp_mprgp,
        time,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. The two algorithms in one paragraph each

        **Active set / block principal pivoting.** Maintain a candidate free set
        $F$. Solve the reduced SPD system $A_{FF}x_F=b_F$ (matrix-free CG), set
        $x_{F^c}=0$, and read the reduced gradient $s=Ax-b$. *Primal violators*
        (free but negative) are dropped to the bound; *dual violators* (bound but
        $s_i<0$) are added to $F$. A whole block is swapped per step, so the optimal
        support is usually found in a **handful of outer steps**, each an exact-ish
        inner solve. A patience counter plus a least-index Bland pivot guarantees
        finite termination with no non-degeneracy assumption.

        **MPRGP.** Stay feasible ($x\ge 0$) throughout. Split the gradient into the
        *free* gradient $\varphi$ (on $\{x_i>0\}$) and the *chopped* gradient
        $\beta$ (the releasing part on $\{x_i=0\}$). The **proportioning test**
        $\lVert\beta\rVert^2 \le \Gamma^2\,\tilde\varphi^\top\varphi$ decides the
        move:

        - proportional → a **CG step** minimising within the current face, or, if it
          would leave the feasible box, an **expansion step** (walk to the bound +
          one fixed-step projected gradient, *adding* constraints);
        - disproportional → a **proportioning step** along $\beta$, *releasing*
          constraints.

        The fixed step $\bar\alpha \in (0, 2/\lVert A\rVert]$ is the only tuning; it
        is estimated matrix-free by power iteration. MPRGP identifies the active set
        in finitely many steps and then *is* CG on the optimal face — so it
        terminates finitely too, with an $R$-linear rate bound along the way.
        """
    )
    return


@app.cell
def _(np):
    # Planted-optimum generator (same construction as tests/problems.make_problem):
    # a known minimiser x* with a chosen support, so we can measure recovery error.
    def make_problem(n, kappa, support_frac, seed):
        rng = np.random.default_rng(seed)
        eig = np.geomspace(1.0, kappa, n)
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        mat = 0.5 * ((q * eig) @ q.T + ((q * eig) @ q.T).T)
        k = max(1, round(support_frac * n))
        perm = rng.permutation(n)
        x_star = np.zeros(n)
        x_star[perm[:k]] = rng.uniform(0.5, 1.5, size=k)
        s_star = np.zeros(n)
        s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
        return mat, mat @ x_star - s_star, x_star

    return (make_problem,)


@app.cell
def _(time):
    # Best-of-`reps` wall-clock timing; returns (seconds, last_result). Best-of
    # rather than mean rejects scheduler/GC noise, the usual microbenchmark hygiene.
    def bench(fn, reps=5):
        best = float("inf")
        result = None
        for _ in range(reps):
            _t = time.perf_counter()
            result = fn()
            best = min(best, time.perf_counter() - _t)
        return best, result

    return (bench,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. They find the same optimum

        First, a sanity check on a single problem: both solvers should land on the
        same point and certify it with a near-zero KKT violation.
        """
    )
    return


@app.cell
def _(DenseOperator, kkt_violation, make_problem, np, solve_nnqp, solve_nnqp_mprgp):
    _a, _b, _x_star = make_problem(120, 1e4, support_frac=0.5, seed=0)
    _op = DenseOperator(_a)

    demo_as = solve_nnqp(_op, _b)
    # tol tightened: MPRGP is first-order, so matching the active-set loop's
    # near-exact iterate at kappa=1e4 needs a tighter projected-gradient tolerance.
    demo_mp = solve_nnqp_mprgp(_op, _b, tol=1e-11)

    demo_agree = float(np.max(np.abs(demo_as.x - demo_mp.x)))
    demo_kkt_as = kkt_violation(_op, _b, demo_as.x)
    demo_kkt_mp = kkt_violation(_op, _b, demo_mp.x)
    return demo_agree, demo_as, demo_kkt_as, demo_kkt_mp, demo_mp


@app.cell
def _(demo_agree, demo_as, demo_kkt_as, demo_kkt_mp, demo_mp, mo):
    _as_steps = f"{demo_as.outer} outer / {demo_as.inner} inner CG"
    _mp_steps = (
        f"{demo_mp.iterations} ({demo_mp.cg_steps} cg, "
        f"{demo_mp.expansion_steps} exp, {demo_mp.proportioning_steps} prop)"
    )
    mo.md(
        rf"""
        | quantity | active set | MPRGP |
        |---|---|---|
        | converged | {demo_as.converged} | {demo_mp.converged} |
        | KKT violation | {demo_kkt_as:.2e} | {demo_kkt_mp:.2e} |
        | steps | {_as_steps} | {_mp_steps} |
        | Hessian products | {demo_as.inner} *(free-block)* | {demo_mp.hessian_products} *(full-size)* |

        **Agreement** $\lVert x_{{\text{{AS}}}} - x_{{\text{{MPRGP}}}}\rVert_\infty
        = {demo_agree:.2e}$ — the same minimiser.

        The step vocabularies differ completely: the active-set loop takes a few
        **outer** steps (each an inner CG solve on the free block), while MPRGP takes
        many small feasible steps. Note the Hessian-product counts are *not*
        directly comparable — the active-set loop's products act on the **reduced**
        block $A_{{FF}}$ (cheaper), MPRGP's on the **full** operator. Wall-clock is
        the fair yardstick, so that is what the scaling study below uses.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Live comparison

        Turn the knobs. We build a planted problem, solve it with both methods at
        **matched accuracy** (MPRGP's tolerance is tightened until its recovery
        error is within $3\times$ the active-set loop's), and time each best-of-5.
        MPRGP's $\bar\alpha$ is precomputed once by power iteration so the estimate
        is not re-charged to every solve.
        """
    )
    return


@app.cell
def _(mo):
    n_slider = mo.ui.slider(50, 600, value=200, step=50, label="dimension $n$")
    kappa_slider = mo.ui.slider(1, 6, value=4, step=1, label="condition number $\\log_{10}\\kappa$")
    supp_slider = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05, label="support fraction")
    seed_slider = mo.ui.slider(0, 20, value=0, step=1, label="seed")
    mo.vstack([n_slider, kappa_slider, supp_slider, seed_slider])
    return kappa_slider, n_slider, seed_slider, supp_slider


@app.cell
def _(
    DenseOperator,
    bench,
    kappa_slider,
    make_problem,
    n_slider,
    np,
    power_iteration,
    seed_slider,
    solve_nnqp,
    solve_nnqp_mprgp,
    supp_slider,
):
    def matched_mprgp(op, b, x_star, target_err, alpha_bar):
        """Tighten MPRGP's tol until recovery error is within 3x the target."""
        tol = 1e-8
        res = solve_nnqp_mprgp(op, b, tol=tol, alpha_bar=alpha_bar)
        for _ in range(8):
            if np.max(np.abs(res.x - x_star)) <= max(target_err, 1e-9) * 3:
                break
            tol *= 0.1
            res = solve_nnqp_mprgp(op, b, tol=tol, alpha_bar=alpha_bar)
        return tol, res

    live_n = n_slider.value
    live_kappa = 10.0**kappa_slider.value
    _a, _b, live_x_star = make_problem(live_n, live_kappa, supp_slider.value, seed_slider.value)
    live_op = DenseOperator(_a)

    live_alpha_bar = 1.0 / float(power_iteration(live_op, seed=0)[0])

    live_t_as, live_as = bench(lambda: solve_nnqp(live_op, _b))
    live_err_as = float(np.max(np.abs(live_as.x - live_x_star)))
    live_tol, _ = matched_mprgp(live_op, _b, live_x_star, live_err_as, live_alpha_bar)
    live_t_mp, live_mp = bench(lambda: solve_nnqp_mprgp(live_op, _b, tol=live_tol, alpha_bar=live_alpha_bar))
    live_err_mp = float(np.max(np.abs(live_mp.x - live_x_star)))
    return (
        live_as,
        live_err_as,
        live_err_mp,
        live_kappa,
        live_mp,
        live_n,
        live_t_as,
        live_t_mp,
    )


@app.cell
def _(live_as, live_err_as, live_err_mp, live_kappa, live_mp, live_n, live_t_as, live_t_mp, mo):
    _speedup = live_t_as / live_t_mp
    _winner = "MPRGP" if _speedup > 1 else "active set"
    mo.md(
        rf"""
        | metric | active set | MPRGP |
        |---|---|---|
        | dimension $n$ | {live_n} | {live_n} |
        | condition number $\kappa$ | {live_kappa:.0e} | {live_kappa:.0e} |
        | wall-clock (best of 5) | **{live_t_as * 1e3:.2f} ms** | **{live_t_mp * 1e3:.2f} ms** |
        | recovery error $\lVert x-x^\star\rVert_\infty$ | {live_err_as:.1e} | {live_err_mp:.1e} |
        | steps | {live_as.outer} outer / {live_as.inner} inner | {live_mp.iterations} iters |

        At these settings the faster solver is **{_winner}**
        (${_speedup:.2f}\times$ active-set / MPRGP). Sweep $\kappa$ up with the
        slider and watch MPRGP fall behind: its iteration count grows like
        $\sqrt\kappa$ while the active-set loop keeps taking only a few outer steps.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Scaling with the condition number

        The decisive axis is conditioning. We fix $n$ and a seed and sweep
        $\kappa$ from $10$ to $10^6$, timing both solvers at matched accuracy. This
        is the plot that answers "*is MPRGP faster?*".
        """
    )
    return


@app.cell
def _(
    DenseOperator,
    bench,
    make_problem,
    np,
    power_iteration,
    solve_nnqp,
    solve_nnqp_mprgp,
):
    sweep_kappa = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    sweep_n = 250
    sweep_seed = 1

    sweep_t_as, sweep_t_mp = [], []
    sweep_prod_as, sweep_prod_mp = [], []
    for _kappa in sweep_kappa:
        _a, _b, _x_star = make_problem(sweep_n, float(_kappa), 0.5, sweep_seed)
        _op = DenseOperator(_a)
        _alpha_bar = 1.0 / float(power_iteration(_op, seed=0)[0])

        _t_as, _r_as = bench(lambda op=_op, b=_b: solve_nnqp(op, b))
        _err_as = np.max(np.abs(_r_as.x - _x_star))

        # match accuracy by tightening MPRGP tol
        _tol = 1e-8
        _r_mp = solve_nnqp_mprgp(_op, _b, tol=_tol, alpha_bar=_alpha_bar)
        for _ in range(8):
            if np.max(np.abs(_r_mp.x - _x_star)) <= max(_err_as, 1e-9) * 3:
                break
            _tol *= 0.1
            _r_mp = solve_nnqp_mprgp(_op, _b, tol=_tol, alpha_bar=_alpha_bar)
        _t_mp, _r_mp = bench(lambda op=_op, b=_b, t=_tol, ab=_alpha_bar: solve_nnqp_mprgp(op, b, tol=t, alpha_bar=ab))

        sweep_t_as.append(_t_as * 1e3)
        sweep_t_mp.append(_t_mp * 1e3)
        sweep_prod_as.append(_r_as.inner)
        sweep_prod_mp.append(_r_mp.hessian_products)
    return (
        sweep_kappa,
        sweep_n,
        sweep_prod_as,
        sweep_prod_mp,
        sweep_t_as,
        sweep_t_mp,
    )


@app.cell
def _(plt, sweep_kappa, sweep_n, sweep_prod_as, sweep_prod_mp, sweep_t_as, sweep_t_mp):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4))

    _ax1.loglog(sweep_kappa, sweep_t_as, "o-", label="active set", color="#11D48E")
    _ax1.loglog(sweep_kappa, sweep_t_mp, "s-", label="MPRGP", color="#555")
    _ax1.set_xlabel(r"condition number $\kappa$")
    _ax1.set_ylabel("wall-clock (ms, best of 5)")
    _ax1.set_title(f"Time vs conditioning ($n={sweep_n}$, matched accuracy)")
    _ax1.legend()
    _ax1.grid(True, which="both", alpha=0.3)

    _ax2.loglog(sweep_kappa, sweep_prod_as, "o-", label="active set (reduced-block CG)", color="#11D48E")
    _ax2.loglog(sweep_kappa, sweep_prod_mp, "s-", label="MPRGP (full-size)", color="#555")
    _ax2.set_xlabel(r"condition number $\kappa$")
    _ax2.set_ylabel("Hessian products")
    _ax2.set_title("Matrix-vector products vs conditioning")
    _ax2.legend()
    _ax2.grid(True, which="both", alpha=0.3)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, np, sweep_kappa, sweep_t_as, sweep_t_mp):
    _ratios = [tm / ta for ta, tm in zip(sweep_t_as, sweep_t_mp, strict=True)]
    _lo, _hi = min(_ratios), max(_ratios)
    _kexp = round(np.log10(sweep_kappa[-1]))
    mo.md(
        rf"""
        **Reading the plots.** On these dense SPD problems the active-set loop is
        as fast or faster across the whole range, and its lead *grows* with
        $\kappa$: MPRGP runs from roughly on-par at low conditioning to
        ${_hi:.1f}\times$ slower at $\kappa=10^{{{_kexp}}}$
        (MPRGP/active-set time ratio spans ${_lo:.2f}$–${_hi:.2f}$).

        Two forces drive this:

        - **Outer-step economy.** The active-set loop reaches the optimal support in
          a few block pivots; each inner CG solve is on the *reduced* block and is
          warm-started. MPRGP's step count grows like $\sqrt\kappa$ (the CG rate),
          and each step touches the *full* operator.
        - **First-order accuracy.** MPRGP never solves a face exactly, so its iterate
          error carries a factor of $\kappa$; matching the active-set loop's
          near-exact answer forces a tighter tolerance and thus *more* steps
          precisely where it is already slower.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. MPRGP's move breakdown

        MPRGP's cost is the sum of three move types. Expansion and proportioning
        steps are the ones that *change* the active set; long runs of CG steps are
        plain conjugate gradients on a fixed face. Watch the CG steps dominate — and
        grow with $\kappa$ — while the constraint-changing steps stay a small,
        roughly constant overhead.
        """
    )
    return


@app.cell
def _(DenseOperator, make_problem, np, power_iteration, solve_nnqp_mprgp):
    break_kappa = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    break_cg, break_exp, break_prop = [], [], []
    for _kappa in break_kappa:
        _a, _b, _x_star = make_problem(250, float(_kappa), 0.5, 1)
        _op = DenseOperator(_a)
        _ab = 1.0 / float(power_iteration(_op, seed=0)[0])
        _r = solve_nnqp_mprgp(_op, _b, tol=1e-10, alpha_bar=_ab)
        break_cg.append(_r.cg_steps)
        break_exp.append(_r.expansion_steps)
        break_prop.append(_r.proportioning_steps)
    return break_cg, break_exp, break_kappa, break_prop


@app.cell
def _(break_cg, break_exp, break_kappa, break_prop, np, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _x = np.arange(len(break_kappa))
    _ax.bar(_x, break_cg, label="CG steps (minimise in face)", color="#11D48E")
    _ax.bar(_x, break_exp, bottom=break_cg, label="expansion (add constraints)", color="#0a8f5f")
    _ax.bar(
        _x,
        break_prop,
        bottom=[c + e for c, e in zip(break_cg, break_exp, strict=True)],
        label="proportioning (release)",
        color="#f0a500",
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f"$10^{{{round(np.log10(k))}}}$" for k in break_kappa])
    _ax.set_xlabel(r"condition number $\kappa$")
    _ax.set_ylabel("MPRGP steps")
    _ax.set_title(r"MPRGP move breakdown ($n=250$, tol $=10^{-10}$)")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. MPRGP is first-order: accuracy tracks the tolerance

        The active-set loop returns a near-exact iterate regardless of `tol` (it
        solves each face by CG to a tight inner tolerance). MPRGP's recovery error,
        by contrast, is proportional to its projected-gradient tolerance times the
        conditioning — halve the tolerance, roughly halve the error. This is the
        price of never factorising a face, and the reason the comparisons above
        matched accuracy explicitly.
        """
    )
    return


@app.cell
def _(DenseOperator, make_problem, np, power_iteration, solve_nnqp_mprgp):
    tol_grid = np.array([1e-6, 1e-8, 1e-10, 1e-12])
    tol_errs = {}
    for _kappa in (1e2, 1e4, 1e6):
        _a, _b, _x_star = make_problem(200, float(_kappa), 0.5, 0)
        _op = DenseOperator(_a)
        _ab = 1.0 / float(power_iteration(_op, seed=0)[0])
        _errs = []
        for _tol in tol_grid:
            _r = solve_nnqp_mprgp(_op, _b, tol=float(_tol), alpha_bar=_ab, max_iter=200_000)
            _errs.append(float(np.max(np.abs(_r.x - _x_star))))
        tol_errs[_kappa] = _errs
    return tol_errs, tol_grid


@app.cell
def _(np, plt, tol_errs, tol_grid):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _colors = {1e2: "#11D48E", 1e4: "#0a8f5f", 1e6: "#f0a500"}
    for _kappa, _errs in tol_errs.items():
        _kexp = round(np.log10(_kappa))
        _ax.loglog(tol_grid, _errs, "o-", label=rf"$\kappa=10^{{{_kexp}}}$", color=_colors[_kappa])
    _ax.set_xlabel(r"MPRGP tolerance (relative projected gradient)")
    _ax.set_ylabel(r"recovery error $\|x - x^\star\|_\infty$")
    _ax.set_title("MPRGP accuracy is proportional to tolerance (and to $\\kappa$)")
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. So — is MPRGP faster? When to reach for each

        **On this problem class, no.** For dense SPD non-negative QPs the
        **active-set loop is the stronger default**: fewer, cheaper effective
        products, machine-accuracy iterates for free, and a lead that widens with
        conditioning. MPRGP ranges from roughly on-par to several times slower here.

        That is not a knock on MPRGP — it is the wrong benchmark for its strengths.
        Reach for **MPRGP** when:

        - **the operator is huge, sparse, or purely matrix-free** (e.g. FEM contact
          problems) — MPRGP never forms or solves a face, so it sidesteps the
          reduced-solve cost that the active-set loop leans on;
        - the active-set search **thrashes combinatorially** (many bound changes,
          the cycling the Bland fallback exists to catch) — MPRGP is monotone with a
          *proven* $R$-linear rate, no worst-case blow-up in the number of faces;
        - a **guaranteed convergence-rate bound** matters more than a machine-exact
          answer, or a good feasible point early is worth more than the last digits.

        Reach for the **active-set loop** (`solve_nnqp`) when the problem is
        moderate and dense-ish, high accuracy is wanted cheaply, or you are
        **warm-starting across a parameter sweep** — a support-stable step converges
        in a single outer iteration, which MPRGP has no direct analogue for.

        The two are complementary, and `nncg` exposes both behind the same operator
        interface so you can swap `solve_nnqp` ↔ `solve_nnqp_mprgp` on the identical
        problem and measure — exactly as this notebook did.

        ### Where to go next

        - **[Active-set methods](01_active_set_methods.html)** — the block-pivot
          machinery and the Bland-fallback termination proof.
        - **[Test problems](03_test_problems.html)** — the planted-optimum
          generators used throughout, including the adversarial family.
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
