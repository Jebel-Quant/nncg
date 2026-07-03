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
        # Active-set methods for non-negative variables

        This notebook explains, in detail, the active-set machinery that `nncg`
        uses to solve the strictly convex **non-negative quadratic program**

        $$
        \min_{x \ge 0}\ \tfrac12\, x^\top A x - b^\top x,
        \qquad A = A^\top \succ 0 .
        $$

        Everything here concerns the bound-only problem solved by
        `nncg.solve_nnqp`. Equality constraints get their own notebook; the
        planted-optimum test problems get a third.

        We build the story bottom-up:

        1. the KKT / complementarity conditions and why they *are* the solution,
        2. the free set / active set and the reduced system,
        3. the primal-dual violator tests that drive the iteration,
        4. block principal pivoting (the fast batch exchange),
        5. why the batch path can cycle and how the **Bland fallback** guarantees
           finite termination,
        6. a live, interactive solve with the free-set trajectory visualised.
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from cvx.linalg import DenseOperator

    from nncg import kkt_violation, solve_nnqp

    return DenseOperator, kkt_violation, np, plt, solve_nnqp


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. The optimality conditions

        The objective $f(x) = \tfrac12 x^\top A x - b^\top x$ is strictly convex
        ($A \succ 0$), so the feasible set $\{x \ge 0\}$ meets it at a **unique**
        global minimiser. Introduce the *reduced gradient*

        $$
        s \;=\; \nabla f(x) \;=\; A x - b .
        $$

        A point $x$ is optimal iff it satisfies the **Karush–Kuhn–Tucker (KKT)**
        system

        $$
        x \ge 0, \qquad s \ge 0, \qquad x \odot s = 0 ,
        $$

        where $\odot$ is the elementwise product. The last equation is
        **complementarity**: for every coordinate $i$, either $x_i = 0$ (the bound
        is *active*) or $s_i = 0$ (the variable is *free* and interior), never both
        strictly positive. This is exactly a **linear complementarity problem**,
        $\mathrm{LCP}(A, -b)$: find $x \ge 0$, $s = Ax - b \ge 0$ with $x^\top s = 0$.

        `nncg` ships this test as a certificate — the maximum violation of the
        three conditions. A value of zero *proves* global optimality.
        """
    )
    return


@app.cell
def _(DenseOperator, kkt_violation, np, solve_nnqp):
    # A tiny, hand-checkable problem: a = diag-ish SPD, b chosen so the optimum
    # keeps some coordinates at the bound and some free.
    _rng = np.random.default_rng(0)
    _q, _ = np.linalg.qr(_rng.standard_normal((6, 6)))
    a_small = (_q * np.geomspace(1.0, 50.0, 6)) @ _q.T
    a_small = 0.5 * (a_small + a_small.T)
    b_small = np.array([2.0, -1.0, 3.0, -0.5, 1.0, 0.4])

    op_small = DenseOperator(a_small)
    res_small = solve_nnqp(op_small, b_small, track=True)

    x = res_small.x
    s = a_small @ x - b_small
    print("x        =", np.round(x, 4))
    print("s = Ax-b  =", np.round(s, 4))
    print("x .* s    =", np.round(x * s, 6), " (complementarity: all ~0)")
    print("KKT violation =", kkt_violation(op_small, b_small, x))
    print("converged     =", res_small.converged)
    return (x,)


@app.cell
def _(mo, x):
    mo.md(
        rf"""
        Read the numbers above coordinate by coordinate: wherever $x_i > 0$ the
        reduced gradient $s_i \approx 0$, and wherever $s_i > 0$ the variable sits
        at its bound $x_i = 0$. That is complementarity in action.

        - **Free set** $F = \{{\, i : x_i > 0 \,\}}$ — the interior variables.
          Here `x > 0` at {[int(i) for i in range(len(x)) if x[i] > 1e-9]}.
        - **Active set** $\mathcal{{A}} = \{{\, i : x_i = 0 \,\}}$ — the bound
          variables, at {[int(i) for i in range(len(x)) if x[i] <= 1e-9]}.

        On the free set the bound is *inactive* and the problem is locally an
        **unconstrained** quadratic — that is the whole idea an active-set method
        exploits.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. The free set and the reduced system

        Suppose we *knew* the optimal free set $F$. Then $x_i = 0$ for $i \notin F$,
        and on $F$ the variables are strictly interior, so the KKT system collapses
        to a plain **unconstrained** stationarity condition on that block:

        $$
        A_{FF}\, x_F \;=\; b_F ,
        \qquad x_{F^c} = 0 .
        $$

        Because $A \succ 0$, every principal submatrix $A_{FF}$ is SPD, so this
        reduced system has a unique solution and can be solved by **conjugate
        gradients** — matrix-free, at the $O(\sqrt{\kappa})$ Krylov rate. This is
        the *inner solve*; `nncg` never forms or factorises $A_{FF}$, it only needs
        the action $v \mapsto A_{FF} v$ (see `nncg.krylov.cg` and the
        `apply_free` / `restricted` operator hooks).

        The catch, of course, is that we **do not** know $F$ in advance. The
        active-set loop is the search for it: guess a free set, solve the reduced
        system, inspect who violates optimality, and correct the guess.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. The two violator tests

        Fix a candidate free set $F$, solve $A_{FF} x_F = b_F$, set $x_{F^c}=0$, and
        form the reduced gradient $s = Ax - b$ over *all* coordinates. Two things
        can be wrong with this guess:

        - **Primal violators** $D = \{\, i \in F : x_i < 0 \,\}$. We *assumed* $i$
          was interior, but the reduced solve pushed it negative — infeasible. Such
          an $i$ must be **dropped** to its bound.

        - **Dual violators** $V = \{\, i \notin F : s_i < 0 \,\}$. We *assumed* $i$
          was pinned at the bound, but its reduced gradient is negative, i.e. the
          objective would *decrease* if we let $x_i$ grow. Such an $i$ must be
          **added** to the free set.

        (Both tests use a tolerance `tol`; `solve_nnqp` defaults to `1e-8`.)

        If **neither** set is non-empty, the current $x$ satisfies $x \ge 0$,
        $s \ge 0$, and complementarity by construction ($x_{F^c}=0$ and, on $F$,
        $s_F = A_{FF}x_F - b_F = 0$). That is the KKT system — we stop, and the
        point is the unique global minimiser. Otherwise we exchange indices and
        repeat. In `nncg.solver._active_set_loop` these are exactly:

        ```python
        s    = reduced_gradient(x, lam)                # s = Ax - b
        prim = np.flatnonzero(free & (x < -tol))       # D: free but negative
        dual = np.flatnonzero((~free) & (s < -tol))    # V: bound but s < 0
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Block principal pivoting — the fast path

        The cheapest way to use the violator sets is a **batch exchange**: drop
        *every* primal violator and add *every* dual violator in a single step,

        $$
        F \;\leftarrow\; \bigl(F \setminus D\bigr) \cup V .
        $$

        This is **block principal pivoting** (BPP): each toggle is a *principal
        pivot* of the LCP, and swapping a whole block at once typically reaches the
        optimal free set in a handful of outer steps — far fewer than the
        one-at-a-time active-set exchanges of classical QP solvers. In the driver:

        ```python
        free[prim] = False   # drop all primal violators D
        free[dual] = True    # add  all dual  violators V
        ```

        Empirically BPP converges in $O(1)$–$O(\log n)$ outer steps on
        well-behaved problems, and each outer step is a single CG solve. That is
        the regime `nncg` lives in almost all of the time.

        But "typically" is not "always" — a pure batch exchange can **cycle**,
        revisiting a free set it has already seen and looping forever. Section 5
        is the fix.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Cycling and the Bland fallback — finite termination

        BPP can over-shoot: on adversarial designs, adding a block of variables
        flips the signs of their partners, which get dropped next step, which flips
        the first block back — a stable loop. To *guarantee* termination `nncg`
        guards the batch path with a **patience counter** and a **least-index
        (Bland-type) fallback**, following Murty's anti-cycling rule for LCPs.

        The bookkeeping is a single scalar $\bar n$ = the best (smallest) number of
        violators seen so far, plus a patience budget `p_max`:

        - If the current violator count $n_{\text{viol}}$ **improves** on $\bar n$,
          take the batch step and refill patience. Real progress is never
          throttled.
        - If it does **not** improve but patience remains, take the batch step
          anyway and spend one unit of patience. (Batch steps are cheap; give them
          a few tries.)
        - If it does not improve **and patience is exhausted**, abandon the batch
          step and take a single **Bland pivot**: toggle only the *least-indexed*
          violator,

          $$
          i^\star = \min\bigl(D \cup V\bigr), \qquad F \leftarrow F \triangle \{i^\star\}.
          $$

        The single least-index pivot is what classical Bland/Murty anti-cycling
        proves cannot repeat a working set, so the loop **cannot cycle** — it
        terminates in finitely many steps at the unique minimiser, with *no
        non-degeneracy assumption* (Theorem 5.1 of the paper). This is the exact
        driver logic:

        ```python
        if n_viol < n_bar or patience > 0:      # fast path
            if n_viol < n_bar:
                n_bar = n_viol; patience = p_max
            else:
                patience -= 1
            free[prim] = False; free[dual] = True
        else:                                   # anti-cycling fallback
            i_star = int(viol.min())
            free[i_star] = not free[i_star]
        ```

        The fallback is not a theoretical ornament: `tests/test_fallback.py`
        exercises it on the `make_adversarial` family, and it is the load-bearing
        component of the termination guarantee.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. A live solve, with the free-set trajectory

        Turn the knobs below. We build a random SPD problem with a *planted*
        optimum of known support (see the problems notebook), solve it with
        `solve_nnqp(..., track=True)`, and plot which coordinates are free at each
        outer step. `track=True` records `Result.traj` — the sequence of visited
        free sets.
        """
    )
    return


@app.cell
def _(mo):
    n_slider = mo.ui.slider(6, 60, value=24, label="dimension $n$")
    kappa_slider = mo.ui.slider(1, 6, value=3, step=1, label="condition number $\\log_{10}\\kappa$")
    supp_slider = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05, label="support fraction")
    seed_slider = mo.ui.slider(0, 30, value=0, step=1, label="seed")
    mo.vstack([n_slider, kappa_slider, supp_slider, seed_slider])
    return kappa_slider, n_slider, seed_slider, supp_slider


@app.cell
def _(kappa_slider, n_slider, np, seed_slider, supp_slider):
    # Planted-optimum generator (same construction as tests/problems.make_problem).
    def make_problem(n, kappa, support_frac, seed):
        rng = np.random.default_rng(seed)
        eig = np.geomspace(1.0, kappa, n)
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        mat = 0.5 * ((q * eig) @ q.T + ((q * eig) @ q.T).T)
        k = max(1, round(support_frac * n))
        perm = rng.permutation(n)
        supp = perm[:k]
        x_star = np.zeros(n)
        x_star[supp] = rng.uniform(0.5, 1.5, size=k)
        s_star = np.zeros(n)
        s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
        return mat, mat @ x_star - s_star, x_star

    n = n_slider.value
    kappa = 10.0**kappa_slider.value
    a, b, x_star = make_problem(n, kappa, supp_slider.value, seed_slider.value)
    return a, b, kappa, n, x_star


@app.cell
def _(DenseOperator, a, b, kkt_violation, solve_nnqp):
    op = DenseOperator(a)
    res = solve_nnqp(op, b, track=True)
    kkt = kkt_violation(op, b, res.x)
    return kkt, res


@app.cell
def _(kappa, kkt, mo, n, np, res, x_star):
    recovered = np.allclose(res.x, x_star, atol=1e-6)
    mo.md(
        rf"""
        | quantity | value |
        |---|---|
        | dimension $n$ | {n} |
        | condition number $\kappa$ | {kappa:.0f} |
        | outer active-set steps | **{res.outer}** |
        | total inner CG iterations | {res.inner} |
        | Bland fallback pivots | {res.fallback} |
        | converged (KKT satisfied) | {res.converged} |
        | KKT violation | {kkt:.2e} |
        | recovered planted optimum | {recovered} |

        Note how few **outer** steps it takes even in high dimension — that is the
        block-pivot fast path. On these random SPD problems the fallback count is
        normally **0**; you need the adversarial family to force it positive.
        """
    )
    return


@app.cell
def _(n, plt, res):
    # Free-set trajectory: rows = outer steps, columns = coordinates.
    # A filled cell means that coordinate was in the free set at that step.
    _traj = res.traj
    _grid = [[1 if i in set(step) else 0 for i in range(n)] for step in _traj]

    _fig, _ax = plt.subplots(figsize=(9, 0.4 * len(_traj) + 1.5))
    _ax.imshow(_grid, aspect="auto", cmap="Greens", interpolation="nearest")
    _ax.set_xlabel("coordinate $i$")
    _ax.set_ylabel("outer step")
    _ax.set_yticks(range(len(_traj)))
    _ax.set_title("Free-set trajectory (green = free / interior at that step)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Each row is one outer iteration; a green cell marks a coordinate that was
        *free* (interior) at that step. Watch the columns settle: after the first
        couple of batch exchanges the free set stops changing — that stationary row
        pattern is the optimal support, and the loop exits on the KKT test the next
        time it finds no violators.

        ### Where to go next

        - **[Equality constraints](02_equality_constraints.html)** — add a linear
          system $Bx = c$ (the simplex / budget constraint), solved on each free
          set through a $p\times p$ Schur complement.
        - **[Test problems](03_test_problems.html)** — the planted-optimum
          generators used above, including the adversarial family that *forces* the
          Bland fallback.
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
