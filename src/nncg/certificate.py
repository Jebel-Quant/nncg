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

    Examples:
        Note that ``a`` must be an operator — a bare array raises ``TypeError``:

        >>> import numpy as np
        >>> from cvx.linalg import DenseOperator
        >>> a = DenseOperator(np.array([[2.0, 0.0], [0.0, 2.0]]))
        >>> b = np.array([2.0, -2.0])

        The minimiser certifies at zero, while the origin does not:

        >>> round(kkt_violation(a, b, np.array([1.0, 0.0])), 12)
        0.0
        >>> kkt_violation(a, b, np.zeros(2)) > 0
        True
    """
    _require_operator(a, b)
    s = a.matvec(x) - b
    # The leading 0.0 is a floor on a quantity that is mathematically non-negative
    # already, and it is there for the sign of zero rather than the magnitude: when
    # every term is zero, `np.max(-s, initial=0.0)` may hand back -0.0 (its reduce
    # path picks a different one of the two equal zeros on linux than on macOS), and
    # `-0.0` compares equal to `0.0` but does not print the same. Python's `max`
    # replaces its running best only on a strict `>`, so the +0.0 seeded here
    # survives and the certificate reports one platform-independent zero.
    return float(
        max(
            0.0,
            np.max(-x, initial=0.0),
            np.max(-s, initial=0.0),
            np.max(np.abs(x * s), initial=0.0),
        )
    )
