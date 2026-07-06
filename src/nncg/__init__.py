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

from .inner import CG, Exact, Jacobi, Nystrom, NystromConfig
from .krylov import KrylovConfig
from .solver import ActiveSetConfig, ActiveSetSolver, InnerSolver, Result, kkt_violation

__all__ = [
    "CG",
    "ActiveSetConfig",
    "ActiveSetSolver",
    "Exact",
    "InnerSolver",
    "Jacobi",
    "KrylovConfig",
    "Nystrom",
    "NystromConfig",
    "Result",
    "kkt_violation",
]

try:
    __version__ = importlib.metadata.version("nncg")
except importlib.metadata.PackageNotFoundError:
    # Package metadata not available (development/editable install)
    __version__ = "0.0.0"
