# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.13",
#     "numpy>=2.0.0",
#     "matplotlib",
#     "scipy>=1.11",
#     "osqp>=1.0",
#     "clarabel>=0.9",
#     "cvx-linalg>=0.9.6",
#     "nncg",
# ]
#
# [tool.uv.sources]
# nncg = { path = "../../..", editable = true }
# ///
#
# The baseline solvers documented here live in `tests/baselines.py` and the
# planted generators in `tests/problems.py` (both outside the installed
# package). They are reproduced inline below so this notebook is self-contained
# and runs anywhere without touching sys.path.
import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Alternative solvers — how `nncg` compares

        `nncg` is not the only way to solve the non-negative quadratic program
        $\min_{x\ge0}\tfrac12 x^\top A x - b^\top x$. This notebook lines it up
        against four established alternatives on the *same* planted-optimum
        problems the test suite uses, so "correct" means **recovering the known
        $x^\star$** and the interesting axis is *cost* — iterations, wall-clock,
        and accuracy.

        | solver | class | handles $Bx=c$? |
        |---|---|---|
        | **`nncg`** | matrix-free CG inside a guarded active-set loop | yes, any $p$ |
        | **OSQP** | operator splitting (ADMM) | yes, any $p$ |
        | **Clarabel** | interior-point (conic) | yes, any $p$ |
        | **Lawson–Hanson** | classical active-set NNLS, CG inner solves | no (bound-only) |
        | **Duchi** | projected gradient, exact simplex projection | **only $p=1$** |

        The last one is special: Duchi et al.'s (2008) $O(n\log n)$ projection is
        onto the simplex $\{x\ge0,\ \mathbf 1^\top x=\beta\}$, so the first-order
        method built on it applies **only** to the single all-ones normalisation
        constraint — the $p=1$ case. That is exactly the fully-invested
        portfolio constraint this family of problems comes from.
        """
    )
    return


@app.cell
def _():
    import clarabel
    import matplotlib.pyplot as plt
    import numpy as np
    import osqp
    from cvx.linalg import DenseOperator
    from scipy import sparse

    from nncg import kkt_violation, solve_nnqp, solve_nnqp_eq
    from nncg.krylov import cg

    return (
        DenseOperator,
        cg,
        clarabel,
        kkt_violation,
        np,
        osqp,
        plt,
        solve_nnqp,
        solve_nnqp_eq,
        sparse,
    )


@app.cell
def _(np):
    # Inlined planted generators from tests/problems.py (self-containment).
    def make_problem(n, kappa, support_frac=0.5, seed=0):
        """Random SPD problem with prescribed condition number and planted optimum."""
        rng = np.random.default_rng(seed)
        eig = np.geomspace(1.0, kappa, n)
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        a = 0.5 * ((q * eig) @ q.T + ((q * eig) @ q.T).T)
        k = max(1, round(support_frac * n))
        perm = rng.permutation(n)
        x_star = np.zeros(n)
        x_star[perm[:k]] = rng.uniform(0.5, 1.5, size=k)
        s_star = np.zeros(n)
        s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
        return a, a @ x_star - s_star, x_star

    def make_simplex_problem(n, kappa, beta=1.0, support_frac=0.5, seed=0):
        """Planted optimum on the scaled simplex {x >= 0, 1^T x = beta} (p=1)."""
        rng = np.random.default_rng(seed)
        eig = np.geomspace(1.0, kappa, n)
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        a = 0.5 * ((q * eig) @ q.T + ((q * eig) @ q.T).T)
        k = max(2, round(support_frac * n))
        perm = rng.permutation(n)
        supp, off = perm[:k], perm[k:]
        x_star = np.zeros(n)
        x_star[supp] = rng.uniform(0.5, 1.5, size=k)
        x_star[supp] *= beta / x_star[supp].sum()
        s_star = np.zeros(n)
        s_star[off] = rng.uniform(0.5, 1.5, size=n - k)
        lam_star = float(rng.standard_normal())
        b = a @ x_star - lam_star * np.ones(n) - s_star
        return a, b, beta, x_star

    return make_problem, make_simplex_problem


@app.cell
def _(cg, clarabel, np, osqp, sparse):
    # Inlined baseline solvers from tests/baselines.py. Each returns a small
    # dict {x, iters, time_s} so the bars below can read them uniformly.
    from time import perf_counter

    def solve_osqp(a, b, b_eq=None, c_eq=None, tol=1e-9):
        """OSQP (ADMM): min 1/2 x'Ax - b'x, x >= 0, optional B x = c."""
        n = a.shape[0]
        p_mat = sparse.triu(sparse.csc_matrix(a)).tocsc()
        blocks, lo, hi = [sparse.eye(n, format="csc")], [np.zeros(n)], [np.full(n, np.inf)]
        if b_eq is not None:
            blocks.append(sparse.csc_matrix(b_eq))
            lo.append(np.asarray(c_eq, float))
            hi.append(np.asarray(c_eq, float))
        t0 = perf_counter()
        prob = osqp.OSQP()
        prob.setup(
            P=p_mat,
            q=-b,
            A=sparse.vstack(blocks, format="csc"),
            l=np.concatenate(lo),
            u=np.concatenate(hi),
            eps_abs=tol,
            eps_rel=tol,
            max_iter=40000,
            polishing=True,
            verbose=False,
        )
        r = prob.solve()
        return {"x": np.asarray(r.x), "iters": int(r.info.iter), "time_s": perf_counter() - t0}

    def solve_clarabel(a, b, b_eq=None, c_eq=None, tol=1e-9):
        """Clarabel (interior point): min 1/2 x'Ax - b'x, x >= 0, optional B x = c."""
        n = a.shape[0]
        p_mat = sparse.triu(sparse.csc_matrix(a)).tocsc()
        g, h, cones = [], [], []
        if b_eq is not None:
            g.append(sparse.csc_matrix(b_eq))
            h.append(np.asarray(c_eq, float))
            cones.append(clarabel.ZeroConeT(b_eq.shape[0]))
        g.append(sparse.csc_matrix(-np.eye(n)))
        h.append(np.zeros(n))
        cones.append(clarabel.NonnegativeConeT(n))
        s = clarabel.DefaultSettings()
        s.verbose, s.tol_gap_abs, s.tol_gap_rel, s.tol_feas = False, tol, tol, tol
        t0 = perf_counter()
        sol = clarabel.DefaultSolver(p_mat, -b, sparse.vstack(g, format="csc"), np.concatenate(h), cones, s).solve()
        return {"x": np.asarray(sol.x), "iters": int(sol.iterations), "time_s": perf_counter() - t0}

    def solve_lawson_hanson(a, b, tol=1e-9, cg_tol=1e-11):
        """Lawson & Hanson active-set NNLS with in-house CG inner solves."""
        n = a.shape[0]
        passive, x, inner = np.zeros(n, bool), np.zeros(n), 0
        t0 = perf_counter()
        for _ in range(3 * n):
            w = b - a @ x
            cand = np.flatnonzero(~passive)
            if cand.size == 0 or float(np.max(w[cand])) <= tol:
                break
            passive[cand[int(np.argmax(w[cand]))]] = True
            while True:
                idx = np.flatnonzero(passive)
                z_p, k = cg(lambda v, i=idx: a[np.ix_(i, i)] @ v, b[idx], tol=cg_tol)
                inner += k
                if z_p.size == 0 or float(np.min(z_p)) > 0.0:
                    x = np.zeros(n)
                    x[idx] = z_p
                    break
                bad = z_p <= 0.0
                alpha = float(np.min(x[idx][bad] / (x[idx][bad] - z_p[bad])))
                step = np.zeros(n)
                step[idx] = z_p - x[idx]
                x = x + alpha * step
                passive[idx[np.abs(x[idx]) <= tol]] = False
                x[~passive] = 0.0
        return {"x": x, "iters": inner, "time_s": perf_counter() - t0}

    def project_simplex(v, beta):
        """Exact Euclidean projection onto {x >= 0, 1^T x = beta} (Duchi 2008)."""
        u = np.sort(v)[::-1]
        css = np.cumsum(u) - beta
        rho = int(np.nonzero(u - css / np.arange(1, v.size + 1) > 0)[0][-1])
        return np.maximum(v - css[rho] / (rho + 1.0), 0.0)

    def solve_duchi(a, b, beta=1.0, tol=1e-9, max_iter=200000):
        """Accelerated (FISTA) projected gradient on the p=1 simplex."""
        n = a.shape[0]
        step = 1.0 / float(np.linalg.eigvalsh(a)[-1])
        x = project_simplex(np.full(n, beta / n), beta)
        y, t_mom, it = x.copy(), 1.0, 0
        t0 = perf_counter()
        while it < max_iter:
            it += 1
            x_new = project_simplex(y - step * (a @ y - b), beta)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom))
            y = x_new + ((t_mom - 1.0) / t_new) * (x_new - x)
            moved = float(np.linalg.norm(x_new - x))
            x, t_mom = x_new, t_new
            if moved <= tol * max(1.0, float(np.linalg.norm(x))):
                break
        return {"x": x, "iters": it, "time_s": perf_counter() - t0}

    return solve_clarabel, solve_duchi, solve_lawson_hanson, solve_osqp


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Bound-only: `nncg` vs OSQP, Clarabel, Lawson–Hanson

        A single planted problem ($n=200$, $\kappa=10^3$). All four recover
        $x^\star$; what differs is the currency each spends. The iteration
        counts are in **different units** — inner CG iterations for `nncg` and
        Lawson–Hanson, outer ADMM sweeps for OSQP, interior-point steps for
        Clarabel — so compare *wall-clock* across solvers and *iterations*
        within a class.
        """
    )
    return


@app.cell
def _(DenseOperator, kkt_violation, make_problem, np, plt, solve_clarabel, solve_lawson_hanson, solve_nnqp, solve_osqp):
    _a, _b, _xs = make_problem(200, 1e3, seed=0)
    _op = DenseOperator(_a)
    _scale = 1.0 + float(np.linalg.norm(_b))

    from time import perf_counter as _pc

    _t0 = _pc()  # nncg's Result carries no clock, so time the call here
    _rn = solve_nnqp(_op, _b)
    _tn = _pc() - _t0
    _results = {
        "nncg": {"iters": _rn.inner, "time_s": _tn, "x": _rn.x},
        "osqp": solve_osqp(_a, _b),
        "clarabel": solve_clarabel(_a, _b),
        "lawson-\nhanson": solve_lawson_hanson(_a, _b),
    }
    _names = list(_results)
    _times = [_results[k]["time_s"] * 1e3 for k in _names]
    _iters = [_results[k]["iters"] for k in _names]
    _errs = [float(np.max(np.abs(_results[k]["x"] - _xs))) for k in _names]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _ax1.bar(_names, _times, color="#11D48E")
    _ax1.set_ylabel("wall-clock [ms]")
    _ax1.set_title("Time to a certified optimum")
    _ax2.bar(_names, _iters, color="#0b7d55")
    _ax2.set_ylabel("native iterations")
    _ax2.set_yscale("log")
    _ax2.set_title("Iterations (mixed units — see text)")
    _fig.tight_layout()

    print("max || x - x_star || per solver:")
    for _k, _e in zip(_names, _errs, strict=False):
        _kkt = kkt_violation(_op, _b, _results[_k]["x"]) / _scale
        print(f"  {_k.replace(chr(10), ' '):14s} {_e:.2e}   KKT {_kkt:.2e}")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        `nncg` and Clarabel are typically fastest; Lawson–Hanson is correct but
        pays for exchanging **one** index per outer step (where `nncg`'s block
        pivots swap many at once), so its CG bill is far higher. OSQP reaches
        machine accuracy after polishing.

        ## The $p=1$ simplex: adding Duchi

        Now the all-ones normalisation $\mathbf 1^\top x = \beta$. `nncg`'s
        equality solver, OSQP and Clarabel take it as a general constraint;
        **Duchi**'s first-order method exploits that the feasible set is exactly
        a simplex and projects onto it in closed form each step.
        """
    )
    return


@app.cell
def _(DenseOperator, make_simplex_problem, np, plt, solve_clarabel, solve_duchi, solve_nnqp_eq, solve_osqp):
    _a, _b, _beta, _xs = make_simplex_problem(200, 1e2, beta=1.0, seed=0)
    _op = DenseOperator(_a)
    _beq, _ceq = np.ones((1, 200)), np.array([_beta])

    from time import perf_counter as _pc

    _t0 = _pc()
    _re = solve_nnqp_eq(_op, _b, _beq, _ceq)
    _tn = _pc() - _t0

    _results = {
        "nncg_eq": {"iters": _re.inner, "time_s": _tn, "x": _re.x},
        "osqp": solve_osqp(_a, _b, b_eq=_beq, c_eq=_ceq),
        "clarabel": solve_clarabel(_a, _b, b_eq=_beq, c_eq=_ceq),
        "duchi": solve_duchi(_a, _b, beta=_beta),
    }
    _names = list(_results)
    _times = [_results[k]["time_s"] * 1e3 for k in _names]
    _iters = [_results[k]["iters"] for k in _names]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _ax1.bar(_names, _times, color="#11D48E")
    _ax1.set_ylabel("wall-clock [ms]")
    _ax1.set_title(r"$p=1$ simplex: time")
    _ax2.bar(_names, _iters, color="#0b7d55")
    _ax2.set_ylabel("native iterations")
    _ax2.set_yscale("log")
    _ax2.set_title("iterations (mixed units)")
    _fig.tight_layout()

    print("simplex recovery (sum should be beta = 1):")
    for _k in _names:
        _x = _results[_k]["x"]
        _err = np.max(np.abs(_x - _xs))
        _nn = bool(np.all(_x >= -1e-9))
        print(f"  {_k:10s} ||x-x*||={_err:.2e}  1^T x={_x.sum():.6f}  x>=0: {_nn}")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Duchi's method needs many cheap projected-gradient steps — its cost
        scales with $\sqrt{\kappa}$ even accelerated — so it is the slowest to
        high accuracy here, but it is the honest first-order baseline and the
        only one that generalises to *huge* $n$ where forming or factoring $A$
        is impossible. Crucially it is confined to $p=1$: with two or more
        equality rows the feasible set is no longer a simplex and the exact
        projection is gone.

        ### Recap

        - **`nncg`** — matrix-free, block pivots, certified finite termination;
          fast and general.
        - **OSQP / Clarabel** — mature general QP solvers; strong references,
          but they form/factor the matrix.
        - **Lawson–Hanson (cg)** — the classical single-exchange active-set
          method; the direct ancestor of the block-pivot loop.
        - **Duchi** — the $p=1$-only first-order method; the scalable,
          projection-based baseline.

        These baselines live in `tests/baselines.py`; the comparison table is
        `tests/benchmarks/compare.py` (run `uv run python -m
        tests.benchmarks.compare`), and `tests/test_baselines.py` asserts every
        one of them recovers the planted $x^\star$.
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
