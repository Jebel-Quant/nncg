"""Non-negative conjugate gradients.

Solves the strictly convex non-negative quadratic program
``min_{x >= 0} 1/2 x^T A x - b^T x`` (and its equality-augmented variant
``B x = c``) by wrapping matrix-free conjugate gradients in a primal-dual
active-set loop with an unconditional finite-termination guarantee — no
non-degeneracy assumption.

Reference implementation of the paper "Non-Negative Conjugate Gradients"
(Schmelzer & Stoll), whose numerical study doubles as this package's test
suite: https://github.com/Jebel-Quant/mean_variance_solvers.
"""

import importlib.metadata

from .krylov import cg, pcg
from .problems import make_adversarial, make_eq_problem, make_problem, make_scaled_problem
from .solver import Result, kkt_violation, solve_nnqp, solve_nnqp_eq

__all__ = [
    "Result",
    "cg",
    "kkt_violation",
    "make_adversarial",
    "make_eq_problem",
    "make_problem",
    "make_scaled_problem",
    "pcg",
    "solve_nnqp",
    "solve_nnqp_eq",
]

try:
    __version__ = importlib.metadata.version("nncg")
except importlib.metadata.PackageNotFoundError:
    # Package metadata not available (development/editable install)
    __version__ = "0.0.0"
