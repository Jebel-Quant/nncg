"""The KKT certificate for the non-negative quadratic program and its shared precondition.

:func:`kkt_violation` scores how far a candidate is from the unique global
minimiser of ``min_{x>=0} 1/2 x'Ax - b'x`` — zero certifies optimality — and is
the load-bearing check the paper's numerical study reports against.
:func:`_require_operator` is the one operator/right-hand-side precondition shared
by the certificate and both :class:`nncg.solver.ActiveSetSolver` entry points.
"""

from __future__ import annotations

import numpy as np
from cvx.linalg import SymmetricOperator, Vector

_NEEDS_OPERATOR = (
    "the quadratic term must be a cvx.linalg.SymmetricOperator: wrap a dense SPD "
    "array in DenseOperator(a), or pass GramOperator(M, ridge) for A = M'M + ridge*I"
)


def _require_operator(a: SymmetricOperator, b: Vector) -> None:
    """Validate that ``a`` is a symmetric operator whose dimension matches ``b``.

    Args:
        a: The quadratic term, expected to be a :class:`cvx.linalg.SymmetricOperator`.
        b: The linear term ``b``.

    Raises:
        TypeError: When ``a`` is not a :class:`cvx.linalg.SymmetricOperator`.
        ValueError: When the operator dimension does not match ``len(b)``.
    """
    if not isinstance(a, SymmetricOperator):
        raise TypeError(_NEEDS_OPERATOR)
    if a.n != len(b):
        msg = f"operator dimension {a.n} does not match len(b) = {len(b)}"
        raise ValueError(msg)


def kkt_violation(a: SymmetricOperator, b: Vector, x: Vector) -> float:
    """Maximum violation of the KKT system of ``min_{x>=0} 1/2 x'Ax - b'x``.

    Args:
        a: The SPD operator ``A`` (a :class:`cvx.linalg.SymmetricOperator`).
        b: The linear term ``b``.
        x: Candidate solution.

    Returns:
        ``max`` of the negativity violations of ``x`` and of the reduced
        gradient ``s = A x - b``, and of the complementarity products
        ``|x_i s_i|``. Zero certifies the unique global minimiser.
    """
    _require_operator(a, b)
    s = a.matvec(x) - b
    return float(
        max(
            np.max(-x, initial=0.0),
            np.max(-s, initial=0.0),
            np.max(np.abs(x * s), initial=0.0),
        )
    )
