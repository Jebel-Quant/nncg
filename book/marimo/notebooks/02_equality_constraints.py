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
        # The solver, with and without equality constraints

        `nncg` exposes two entry points that share one active-set driver:

        - `solve_nnqp(a, b)` — the **bound-only** program
          $\min_{x \ge 0} \tfrac12 x^\top A x - b^\top x$.
        - `solve_nnqp_eq(a, b, b_eq, c_eq)` — the same objective **plus** a linear
          equality system $Bx = c$.

        The active-set notebook covered the bound-only case in detail. Here we
        focus on what the equality constraints add: a **multiplier** $\lambda$, a
        **saddle system** on each free set, and a $p \times p$ **Schur complement**
        that eliminates $\lambda$ using the same matrix-free CG inner solver. The
        important structural fact is that *the outer loop does not change at all* —
        only the per-free-set subproblem does.

        (The code uses the library's own argument names: `a` for the SPD matrix
        $A$, `b_eq` for the equality matrix $B$, `c_eq` for its right-hand side.)
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
def _(mo):
    mo.md(
        r"""
        ## 1. Without equality constraints (recap)

        For the bound-only problem the KKT system is the LCP

        $$
        x \ge 0, \qquad s = Ax - b \ge 0, \qquad x \odot s = 0 .
        $$

        On a fixed free set $F$ the subproblem is a single reduced SPD solve
        $A_{FF} x_F = b_F$, done matrix-free by CG. That is the whole per-step cost.
        """
    )
    return


@app.cell
def _(DenseOperator, kkt_violation, np, solve_nnqp):
    _rng = np.random.default_rng(0)
    _q, _ = np.linalg.qr(_rng.standard_normal((80, 80)))
    a = (_q * np.geomspace(1.0, 1e3, 80)) @ _q.T
    a = 0.5 * (a + a.T)
    b = _rng.standard_normal(80)

    op = DenseOperator(a)
    res = solve_nnqp(op, b)
    print("bound-only  solve_nnqp")
    print("  outer steps        :", res.outer)
    print("  inner CG iterations:", res.inner)
    print("  converged          :", res.converged)
    print("  KKT violation      :", kkt_violation(op, b, res.x))
    print("  # active (x==0)    :", int((res.x <= 1e-9).sum()), "of", len(b))
    print("  sum(x)             :", round(float(res.x.sum()), 4), " (unconstrained)")
    return a, b


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Adding equality constraints

        Now constrain the same problem with $B x = c$, where $B \in \mathbb{R}^{p
        \times n}$ has full row rank:

        $$
        \min_{x \ge 0}\ \tfrac12 x^\top A x - b^\top x
        \quad \text{s.t.}\quad B x = c .
        $$

        Attach a multiplier $\lambda \in \mathbb{R}^p$ to the equalities. The
        Lagrangian is $\tfrac12 x^\top A x - b^\top x - \lambda^\top (Bx - c)$, and
        the **constrained reduced gradient** becomes

        $$
        s \;=\; A x - b - B^\top \lambda .
        $$

        The KKT / complementarity conditions read exactly as before *but in this
        shifted $s$*, together with primal feasibility of the equalities:

        $$
        x \ge 0,\quad s \ge 0,\quad x \odot s = 0,\qquad B x = c .
        $$

        The multiplier $\lambda$ is **free** (no sign restriction) — it is
        determined, not searched over. The active-set loop still toggles only the
        *bound* constraints; the equalities are enforced exactly at every step.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. The saddle system on a free set

        Fix a free set $F$. With $x_{F^c}=0$, stationarity on $F$ plus the
        equalities give the **saddle (KKT) system**

        $$
        \begin{bmatrix} A_{FF} & B_F^\top \\ B_F & 0 \end{bmatrix}
        \begin{bmatrix} x_F \\ -\lambda \end{bmatrix}
        =
        \begin{bmatrix} b_F \\ c \end{bmatrix},
        $$

        where $B_F$ is $B$ restricted to the free columns. This is indefinite, so
        we do **not** hand it to CG directly. Instead we eliminate $\lambda$ with a
        **Schur complement**. From the first block,
        $x_F = A_{FF}^{-1}(b_F + B_F^\top \lambda)$. Substituting into $B_F x_F = c$:

        $$
        \underbrace{B_F A_{FF}^{-1} B_F^\top}_{S \,\in\, \mathbb{R}^{p\times p}}\,
        \lambda \;=\; c - B_F A_{FF}^{-1} b_F .
        $$

        $S$ is $p \times p$ and **SPD** (because $A_{FF}\succ 0$ and $B_F$ has full
        row rank), so a tiny Cholesky solve fixes $\lambda$; then $x_F$ follows in
        closed form. Crucially, forming $S$ and the right-hand side needs only

        $$
        v_0 = A_{FF}^{-1} b_F, \qquad v_1^{(j)} = A_{FF}^{-1} (B_F^\top e_j),\ j=1..p,
        $$

        i.e. **$p+1$ SPD solves that all share the operator $A_{FF}$** — each one a
        matrix-free CG call. So an equality-augmented step costs $p+1$ CG solves
        instead of one, and the matrix is still never formed. This is exactly
        `sub_solve` inside `solve_nnqp_eq`:

        ```python
        v0, k0 = cg(matvec_f, b[idx], tol=cg_tol, x0=x0)     # A_F^{-1} b_F
        v1 = np.zeros((idx.size, p))
        for j in range(p):
            v1[:, j], kj = cg(matvec_f, b_f[j], tol=cg_tol)  # A_F^{-1} B_F^T e_j
        schur = b_f @ v1                                     # S = B_F A_F^{-1} B_F^T
        lam   = cholesky_solve(schur, c_eq - b_f @ v0)       # S lam = c - B_F v0
        xf    = v0 + v1 @ lam                                # x_F
        ```
        """
    )
    return


@app.cell
def _(DenseOperator, a, b, np, solve_nnqp_eq):
    # Add p equality constraints to the SAME (a, b) as the bound-only solve.
    _rng = np.random.default_rng(7)
    p = 3
    b_eq = _rng.standard_normal((p, len(b)))
    # Pick c that is definitely reachable by some x >= 0 (use a random x0 >= 0).
    _x0 = np.abs(_rng.standard_normal(len(b)))
    c = b_eq @ _x0

    op_eq = DenseOperator(a)
    res_eq = solve_nnqp_eq(op_eq, b, b_eq, c)

    print("equality-augmented  solve_nnqp_eq   (p =", p, ")")
    print("  outer steps        :", res_eq.outer)
    print("  inner CG iterations:", res_eq.inner, " (~", p + 1, "x the bound-only work per step)")
    print("  converged          :", res_eq.converged)
    print("  multipliers  lam   :", np.round(res_eq.lam, 4))
    print("  || B x - c ||      :", float(np.linalg.norm(b_eq @ res_eq.x - c)))
    print("  min(x)             :", float(res_eq.x.min()), " (feasibility x >= 0)")
    return b_eq, c, p, res_eq


@app.cell
def _(b_eq, c, mo, np, p, res_eq):
    mo.md(
        rf"""
        The solve returns a multiplier vector `lam` of shape `({p},)` and the
        equality residual $\lVert Bx - c\rVert$ is
        ${np.linalg.norm(b_eq @ res_eq.x - c):.2e}$ — the constraint is satisfied to
        solver tolerance. The dual test in the outer loop now uses the **shifted**
        reduced gradient $s = Ax - b - B^\top \lambda$; `reduced_gradient` in
        `solve_nnqp_eq` adds precisely that $-B^\top\lambda$ correction. Everything
        else — the primal/dual violator tests, the batch exchange, the Bland
        fallback — is the *identical* driver `_active_set_loop` used by the
        bound-only solver.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. The $p = 1$ case: a budget / simplex constraint

        The single most common equality is the **normalisation** $\mathbf{1}^\top x
        = \beta$ (with $B = \mathbf{1}^\top$, $c = \beta$). Combined with $x \ge 0$
        this is the **scaled simplex** — the feasible set of a long-only portfolio
        with a fixed budget, of a mixture weight vector, of any convex combination.
        Here $p = 1$, so the Schur complement is a **scalar** and $B_F$ trivially
        has full row rank on any non-empty free set. This is the workhorse case and
        needs no special handling — it is just `solve_nnqp_eq` with a one-row $B$.

        Slide the budget $\beta$ below and watch the solution renormalise while
        staying non-negative.
        """
    )
    return


@app.cell
def _(mo):
    beta_slider = mo.ui.slider(0.2, 5.0, value=1.0, step=0.1, label=r"budget $\beta = \mathbf{1}^\top x$")
    beta_slider
    return (beta_slider,)


@app.cell
def _(DenseOperator, a, b, beta_slider, np, solve_nnqp_eq):
    beta = beta_slider.value
    b_eq1 = np.ones((1, len(b)))
    c1 = np.array([beta])
    res_b = solve_nnqp_eq(DenseOperator(a), b, b_eq1, c1)
    return beta, res_b


@app.cell
def _(beta, mo, res_b):
    mo.md(
        rf"""
        | quantity | value |
        |---|---|
        | budget $\beta$ | {beta:.2f} |
        | $\sum_i x_i$ (should equal $\beta$) | **{res_b.x.sum():.4f}** |
        | $\min_i x_i$ (feasibility) | {res_b.x.min():.2e} |
        | # active bounds ($x_i = 0$) | {int((res_b.x <= 1e-9).sum())} of {len(res_b.x)} |
        | scalar multiplier $\lambda$ | {float(res_b.lam[0]):.4f} |
        | outer steps | {res_b.outer} |

        The mass always sums to $\beta$ and never goes negative — the constraint
        surface is the simplex $\{{x \ge 0,\ \mathbf1^\top x = \beta\}}$, and the
        support (the non-zero coordinates) shifts as $\beta$ changes.
        """
    )
    return


@app.cell
def _(plt, res_b):
    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.bar(range(len(res_b.x)), res_b.x, color="#11D48E")
    _ax.axhline(0.0, color="k", lw=0.8)
    _ax.set_xlabel("coordinate $i$")
    _ax.set_ylabel("$x_i$")
    _ax.set_title(rf"Optimal weights on the simplex ($\sum x_i = {res_b.x.sum():.2f}$)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Cost comparison: bound-only vs. $p$ constraints

        Because each equality-augmented outer step solves $p+1$ CG systems that
        share $A_{FF}$, the **inner** work scales roughly like $(p+1)\times$ the
        bound-only work per outer step, while the **outer** step count stays in the
        same regime. The panel below sweeps $p$ on one fixed $(A, b)$ and a random
        full-rank $B$.
        """
    )
    return


@app.cell
def _(DenseOperator, np, plt, solve_nnqp, solve_nnqp_eq):
    _rng = np.random.default_rng(3)
    _n = 120
    _q, _ = np.linalg.qr(_rng.standard_normal((_n, _n)))
    _core = (_q * np.geomspace(1.0, 1e3, _n)) @ _q.T
    a2 = 0.5 * (_core + _core.T)
    b2 = _rng.standard_normal(_n)
    op2 = DenseOperator(a2)

    ps = [0, 1, 2, 4, 8]
    outers, inners = [], []
    for _p in ps:
        if _p == 0:
            _r = solve_nnqp(op2, b2)
        else:
            _b_eq = _rng.standard_normal((_p, _n))
            _x0 = np.abs(_rng.standard_normal(_n))
            _r = solve_nnqp_eq(op2, b2, _b_eq, _b_eq @ _x0)
        outers.append(_r.outer)
        inners.append(_r.inner)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    _a1.plot(ps, outers, "o-", color="#11D48E")
    _a1.set_xlabel("number of equality constraints $p$")
    _a1.set_ylabel("outer active-set steps")
    _a1.set_title("Outer steps stay flat")
    _a2.plot(ps, inners, "o-", color="#0b7d55")
    _a2.set_xlabel("number of equality constraints $p$")
    _a2.set_ylabel("total inner CG iterations")
    _a2.set_title(r"Inner work grows $\sim (p{+}1)\times$")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Summary

        | | `solve_nnqp` | `solve_nnqp_eq` |
        |---|---|---|
        | feasible set | $x \ge 0$ | $x \ge 0,\ Bx = c$ |
        | reduced gradient | $s = Ax - b$ | $s = Ax - b - B^\top\lambda$ |
        | per-free-set subproblem | one SPD solve $A_{FF}x_F = b_F$ | saddle system via $p\times p$ Schur complement |
        | inner CG solves per outer step | 1 | $p + 1$ (shared operator $A_{FF}$) |
        | outer loop | `_active_set_loop` | **same** `_active_set_loop` |
        | extra output | — | multipliers `Result.lam` |

        The equality machinery is entirely contained in the per-free-set
        subproblem. The termination guarantee — batch pivots guarded by a
        least-index Bland fallback — is inherited unchanged.

        Next: **[Test problems](03_test_problems.html)** documents the
        planted-optimum generators (including the equality-augmented
        `make_eq_problem`) that these demos and the paper's numerical study are
        built on.
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
