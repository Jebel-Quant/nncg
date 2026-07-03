<div align="center" markdown="1">

# ➕ nncg — Non-Negative Conjugate Gradients

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

## 📦 Installation

```bash
pip install nncg
```

## 🚀 Quickstart

```python
import numpy as np
from cvx.linalg import DenseOperator, GramOperator
from nncg import solve_nnqp, solve_nnqp_eq, make_problem

# a planted problem whose optimum is known in closed form
A, b, x_star, _ = make_problem(n=200, kappa=1e4, seed=0)
op = DenseOperator(A)                     # the solvers take a SymmetricOperator

res = solve_nnqp(op, b)
assert res.converged                      # KKT certificate reached
assert np.allclose(res.x, x_star, atol=1e-6)
assert res.outer < 10                     # single-digit outer steps

# equality-augmented: minimise subject to x >= 0 and B x = c
B = np.ones((1, 200))                     # p = 1: the budget 1'x = 1
res_eq = solve_nnqp_eq(op, b, B, np.array([1.0]))
assert res_eq.lam.shape == (1,)           # multiplier, via a p-by-p Schur solve

# warm-start a parametric sweep: support-stable steps take ONE outer step
res2 = solve_nnqp(op, b + 1e-4, warm=(res.free, res.x))

# Gram-structured: A = M'M + I only through products with M — never formed
M = np.random.default_rng(1).standard_normal((50, 200))
res_g = solve_nnqp(GramOperator(M, ridge=1.0), M.T @ np.ones(50))
assert res_g.converged
```

## 🔬 The algorithm in one paragraph

Fix a working set of *free* variables and solve the unconstrained reduced SPD
system by CG (matrix-free, $O(\sqrt{\kappa})$ Krylov rate). Push any free
variable that returns negative to its bound (primal step); release any bound
variable whose reduced gradient is negative (dual step); repeat. Batch
exchanges are fast but can cycle; a patience counter falls back to Murty's
least-index single pivot, which cannot — hence finite termination without any
non-degeneracy hypothesis, and the fallback is provably necessary: on
anti-correlated designs (`make_adversarial`) the unguarded batch path revisits
a previously seen working set and loops forever.

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
