# nncg

Non-negative conjugate gradients: the strictly convex bound-constrained
quadratic `min_{x>=0} 1/2 x'Ax - b'x` (and its equality-augmented variant
`Bx = c`) solved by matrix-free CG inside a primal-dual active-set loop with
an unconditional finite-termination guarantee.

Reference implementation of the paper *Non-Negative Conjugate Gradients*
(Schmelzer & Stoll) — see the
[paper repository](https://github.com/Jebel-Quant/mean_variance_solvers).

## Install

```bash
pip install nncg
```

## API

The public API is `solve_nnqp`, `solve_nnqp_eq`, `cg`, `pcg`,
`kkt_violation`, the `Result` dataclass, and the planted-problem generators
`make_problem`, `make_eq_problem`, `make_adversarial`, `make_scaled_problem`.
See the README for a quickstart.
