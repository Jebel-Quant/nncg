"""Tests extracted directly from the paper "Non-Negative Conjugate Gradients".

Each module pins a claim of the paper (Schmelzer & Stoll,
Jebel-Quant/mean_variance_solvers, ``non_negative_cg/``) that the ``tests.test_nncg``
suite does not already cover, and names the proposition / lemma / result it
verifies:

- ``test_cg_convergence`` — the inner-solve guarantees: the Chebyshev
  energy-norm bound (Prop. 3.1), the ideal-preconditioner one-step collapse
  (Prop. 6.2 / ``prop:pcg``) and the inexact-solve error bound (Lemma 5.2).
- ``test_reduction``      — the Schur-complement elimination of the equality
  multipliers (Sec. 3): agreement with the indefinite saddle solve and the
  pure-normalisation closed form.
- ``test_conditioning``   — the regularising split's condition-number bound and
  its monotone decrease in the split weight (Prop. 6.1).
- ``test_results``        — the numerical study (Sec. 7): outer count small and
  fallback dormant across the ``kappa`` range, inner count under the
  ``sqrt(kappa)`` envelope, regularisation buying the termination guarantee on
  a rank-deficient operator, and the matrix-free deblurring recovery.
"""
