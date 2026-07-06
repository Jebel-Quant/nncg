<div align="center" markdown="1">

# ➕ nncg — Non-Negative Conjugate Gradients

[![CI](https://github.com/Jebel-Quant/nncg/actions/workflows/rhiza_ci.yml/badge.svg)](https://github.com/Jebel-Quant/nncg/actions/workflows/rhiza_ci.yml)
[![Coverage](https://jebel-quant.github.io/nncg/coverage-badge.svg)](https://jebel-quant.github.io/nncg/reports/html-coverage/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/Jebel-Quant/nncg/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CodeQL](https://github.com/Jebel-Quant/nncg/actions/workflows/rhiza_codeql.yml/badge.svg)](https://github.com/Jebel-Quant/nncg/actions/workflows/rhiza_codeql.yml)
[![Rhiza](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2FJebel-Quant%2Fnncg%2Fmain%2F.rhiza%2Ftemplate.yml&query=%24.ref&label=rhiza)](https://github.com/jebel-quant/rhiza)
[![Paper](https://img.shields.io/badge/paper-Non--Negative_Conjugate_Gradients-red?logo=adobeacrobatreader)](https://github.com/Jebel-Quant/mean_variance_solvers)

---

**Quick Links:**
[📄 Paper](https://github.com/Jebel-Quant/mean_variance_solvers) •
[🐛 Report Bug](https://github.com/Jebel-Quant/nncg/issues) •
[💡 Request Feature](https://github.com/Jebel-Quant/nncg/issues)

---

</div>

## 📋 Overview

`nncg` solves the strictly convex non-negative quadratic program

$$\min_{x \geq 0}\ \tfrac{1}{2} x^\top A x - b^\top x, \qquad A \succ 0,$$

and its equality-augmented variant with a general linear system $Bx = c$, by
wrapping **matrix-free conjugate gradients** in a **primal-dual active-set
loop**. The working-set toggles are the principal pivots of the linear
complementarity problem $\mathrm{LCP}(A, -b)$; guarding the fast block-pivot
path with a least-index Bland fallback gives **unconditional finite
termination** at the unique global minimiser — no non-degeneracy assumption.

This is the reference implementation of the paper *Non-Negative Conjugate
Gradients* (Schmelzer & Stoll), developed in
[Jebel-Quant/mean_variance_solvers](https://github.com/Jebel-Quant/mean_variance_solvers).
The paper's numerical study doubles as this package's test suite: planted-optimum
recovery across condition numbers, the equality-augmented solve for
$p \in \{1, 3, 8\}$, CG-vs-exact free-set trajectory agreement (the inexactness
lemma), warm-started parameter sweeps, and the adversarial anti-correlated
family on which the *unguarded* batch path provably cycles and the fallback
terminates.

The quadratic term enters as a `cvx.linalg.SymmetricOperator`: wrap an
explicit SPD array in `DenseOperator`. When $A = M^\top M$ is a Gram matrix,
pass `GramOperator(M, ridge)` and the inner solves need only products with
$M$ — the $n \times n$ matrix is never formed and working memory is $O(n)$.

Each free-block solve is delegated to a **pluggable inner solver** — plain CG
(`CG`), Jacobi- or randomized-Nyström-preconditioned CG (`Jacobi`, `Nystrom`),
or a direct factorisation (`Exact`) — so you match the inner solve to the
operator's structure without touching the outer loop. `ActiveSetSolver` owns the
loop and knows nothing about preconditioning; new inner solvers plug in by
implementing a one-method `InnerSolver` interface.

## 📦 Installation

```bash
pip install nncg
```

## 🚀 Quickstart

The one-call `solve_nnqp` / `solve_nnqp_eq` wrappers cover the common case —
pass a plain SPD array and name the inner solver as a string:

```python
import numpy as np
from nncg import solve_nnqp, solve_nnqp_eq

# a random SPD problem with condition number 1e4
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.standard_normal((200, 200)))
A = (Q * np.geomspace(1.0, 1e4, 200)) @ Q.T
b = rng.standard_normal(200)

res = solve_nnqp(A, b, inner="cg")         # inner solver: "cg" / "jacobi" / "nystrom" / "exact"
assert res.converged                       # stopped on the KKT certificate

# equality-augmented: minimise subject to x >= 0 and B x = c
B = np.ones((1, 200))                      # p = 1: the budget 1'x = 1
res_eq = solve_nnqp_eq(A, b, B, np.array([1.0]), inner="jacobi")
assert res_eq.lam.shape == (1,)            # multiplier, via a p-by-p Schur solve
```

For reuse across a parametric sweep, a matrix-free Gram operator, or a tuned
inner solver, build the `ActiveSetSolver` and its operator directly — the
wrappers are logic-free shortcuts to exactly this:

```python
from cvx.linalg import DenseOperator, GramOperator
from nncg import ActiveSetSolver, CG, Jacobi, Nystrom, NystromConfig, kkt_violation

op = DenseOperator(A)                       # kkt_violation takes a SymmetricOperator too
solver = ActiveSetSolver(inner=CG())        # configure once, reuse across problems
res = solver.solve(op, b)
assert kkt_violation(op, b, res.x) < 1e-6   # zero certifies the global minimiser

# warm-start a parametric sweep: support-stable steps take ONE outer step
res2 = solver.solve(op, b + 1e-4, warm=(res.free, res.x))

# Gram-structured: A = M'M + I only through products with M — never formed.
# Swap the inner solver freely — here Jacobi to strip the diagonal scaling.
M = rng.standard_normal((50, 200))
res_g = ActiveSetSolver(inner=Jacobi()).solve(GramOperator(M, ridge=1.0), M.T @ np.ones(50))
assert res_g.converged

# tuned inner solver: pass the instance (the string shortcut takes defaults only)
res_n = ActiveSetSolver(inner=Nystrom(nystrom=NystromConfig(rank=20))).solve(op, b)
```

## 🔬 The algorithm in one paragraph

Fix a working set of *free* variables and solve the unconstrained reduced SPD
system by CG (matrix-free, $O(\sqrt{\kappa})$ Krylov rate). Push any free
variable that returns negative to its bound (primal step); release any bound
variable whose reduced gradient is negative (dual step); repeat. Batch
exchanges are fast but can cycle; a patience counter falls back to Murty's
least-index single pivot, which cannot — hence finite termination without any
non-degeneracy hypothesis, and the fallback is provably necessary: on
anti-correlated designs (the `make_adversarial` family in the test suite's
`tests/problems.py`) the unguarded batch path revisits a previously seen
working set and loops forever.

## 📖 Citation

If you use this package in academic work, please cite the paper:

```bibtex
@techreport{schmelzer2026nncg,
  title       = {Non-Negative Conjugate Gradients},
  author      = {Schmelzer, Thomas and Stoll, Martin},
  year        = {2026},
  institution = {Jebel Quant Research and TU Chemnitz},
  url         = {https://github.com/Jebel-Quant/mean_variance_solvers},
}
```

## ⚖️ License

MIT — see [LICENSE](LICENSE).
