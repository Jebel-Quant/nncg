"""Inner solvers: one free-block system ``A[F, F] y = rhs`` per active-set step.

Each concrete inner solver provides ``solve(op, idx, rhs, x0) -> (y, iters)``,
solving the free-block system ``A[F, F] y = rhs`` (and so satisfies the
:class:`nncg.solver.InnerSolver` interface). This is the only module that knows
about preconditioning: the built-in solvers are the identity/Jacobi/Nyström-
preconditioned CG variants (:class:`CG`, :class:`Jacobi`, :class:`Nystrom`,
:class:`GlobalNystrom`) and the direct :class:`Exact`. The operator-derived
builders they run on — the free-block matvec and the diagonal/Nyström
preconditioners — live in :mod:`nncg.preconditioners`. Further inner solvers —
e.g. Clarabel- or KKT-equation-based — live in Jebel-Quant/mean_variance_solvers
and satisfy the same structural interface.

Examples:
    The inner solver is the one thing that varies between these runs — the outer
    loop, and the minimiser it certifies, are the same:

    >>> import numpy as np
    >>> from cvx.linalg import DenseOperator
    >>> from nncg import CG, ActiveSetSolver, Exact, Jacobi
    >>> a = DenseOperator(np.diag([1.0, 2.0, 4.0, 8.0]))
    >>> b = np.array([1.0, 2.0, -4.0, -8.0])
    >>> for inner in (CG(), Jacobi(), Exact()):
    ...     res = ActiveSetSolver(inner=inner).solve(a, b)
    ...     print(type(inner).__name__, res.converged, np.allclose(res.x, [1.0, 1.0, 0.0, 0.0]))
    CG True True
    Jacobi True True
    Exact True True

    The direct solver counts one inner "iteration" per solve, so it never spends
    more than the number of outer steps:

    >>> direct = ActiveSetSolver(inner=Exact()).solve(a, b)
    >>> direct.inner <= direct.outer
    True

    The Nyström solvers sketch the free block at :attr:`NystromConfig.rank`, so
    they belong on a problem larger than that rank and with a decaying spectrum
    — here three orders of geometric decay, with a planted optimum on the even
    coordinates:

    >>> from nncg import GlobalNystrom, Nystrom
    >>> d = 10.0 ** -np.linspace(0.0, 3.0, 40)
    >>> x_star = np.where(np.arange(40) % 2 == 0, 1.0, 0.0)
    >>> b = d * x_star - (1.0 - x_star)
    >>> for inner in (Nystrom(), GlobalNystrom()):
    ...     res = ActiveSetSolver(inner=inner).solve(DenseOperator(np.diag(d)), b)
    ...     print(type(inner).__name__, res.converged, np.allclose(res.x, x_star))
    Nystrom True True
    GlobalNystrom True True
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from cvx.linalg import SymmetricOperator, Vector
from numpy.typing import NDArray

from .krylov import KrylovConfig, Preconditioner, pcg
from .preconditioners import (
    GlobalNystromSketch,
    NystromConfig,
    _free_matvec,
    _global_nystrom_sketch,
    _jacobi,
    _masked_nystrom,
    _nystrom,
)

__all__ = ["CG", "Exact", "GlobalNystrom", "Jacobi", "Nystrom", "NystromConfig"]

_RCOND_MIN = 1e-12  # matches cvx-linalg's DEFAULT_COND_THRESHOLD of 1e12


def _default_krylov() -> KrylovConfig:
    """Inner CG config with ``tol`` two orders below the outer tolerance (Lemma 5.1)."""
    return KrylovConfig(tol=1e-10)


def _pcg_block(
    op: SymmetricOperator,
    idx: NDArray[np.int_],
    rhs: Vector,
    x0: Vector | None,
    krylov: KrylovConfig,
    precond: Preconditioner | None,
) -> tuple[Vector, int]:
    """Solve the free block ``A[F, F] y = rhs`` by (preconditioned) CG, warm-started at ``x0``."""
    return pcg(_free_matvec(op, idx), rhs, replace(krylov, precond=precond, x0=x0))


def _same_free_block(
    checked_op: SymmetricOperator | None,
    checked_idx: NDArray[np.int_] | None,
    op: SymmetricOperator,
    idx: NDArray[np.int_],
) -> bool:
    """Return whether ``(op, idx)`` is the memoised free block already verified.

    Keyed on operator *identity* (not equality) so the memo can never carry a
    stale verdict across operators; ``checked_idx is None`` means nothing has
    been verified yet.
    """
    return checked_op is op and checked_idx is not None and np.array_equal(checked_idx, idx)


def _raise_if_singular(op: SymmetricOperator, idx: NDArray[np.int_]) -> None:
    """Raise if the free block ``A[F, F]`` is numerically singular (``rcond_free < _RCOND_MIN``)."""
    rcond = op.rcond_free(idx)
    if rcond < _RCOND_MIN:
        msg = f"free block of size {idx.size} is numerically singular (rcond={rcond:.2e})"
        raise ValueError(msg)


@dataclass(frozen=True)
class Exact:
    """Direct free-block solve via ``op.solve_free`` (one "iteration" per solve).

    Suits backends whose ``solve_free`` is structured and cheap (e.g.
    ``FactorOperator``'s Woodbury solve at ``O(|F| r^2)``). It ignores warm starts.

    The ``rcond_free`` conditioning guard depends only on the free block, not the
    right-hand side, so it is estimated at most once per free set:
    :meth:`nncg.solver.ActiveSetSolver.solve_eq` drives ``p + 1`` solves through
    the *same* free set per outer step, and the (up to ``O(|F|^3)``) estimate must
    not be paid ``p + 1`` times over. The last verified ``(operator, idx)`` is
    memoised in a private single slot — keyed on operator identity so the memo can
    never carry a stale verdict across operators, and excluded from equality/repr
    so ``Exact`` stays a value.

    On the plain :meth:`~nncg.solver.ActiveSetSolver.solve` path every outer step
    visits a *different* free set, so the memo never hits and the guard is paid on
    every step — where, for a dense free block, the ``O(|F|^3)`` eigendecomposition
    can cost several times the Cholesky solve it precedes. The guard is also
    redundant when ``solve_free`` already fails loudly on a rank-deficient block
    (e.g. ``cvx.linalg.cholesky_solve``'s Cholesky→LU fallback). Set
    ``check_conditioning=False`` to skip it and let ``solve_free`` surface any
    singularity itself.

    Attributes:
        check_conditioning: Estimate ``rcond_free`` and raise on a numerically
            singular free block before each (new) solve. Default ``True``; set
            ``False`` to trade the diagnostic for the raw structured solve.
    """

    check_conditioning: bool = True
    _checked_op: SymmetricOperator | None = field(default=None, compare=False, repr=False)
    _checked_idx: NDArray[np.int_] | None = field(default=None, compare=False, repr=False)

    def solve(self, op: SymmetricOperator, idx: NDArray[np.int_], rhs: Vector, x0: Vector | None) -> tuple[Vector, int]:  # noqa: ARG002
        """Solve the free block ``A[F, F] y = rhs`` directly, guarding its conditioning once per free set."""
        if self.check_conditioning and not _same_free_block(self._checked_op, self._checked_idx, op, idx):
            _raise_if_singular(op, idx)
            object.__setattr__(self, "_checked_op", op)
            object.__setattr__(self, "_checked_idx", idx)
        return op.solve_free(idx, rhs), 1


@dataclass(frozen=True)
class CG:
    """Plain matrix-free conjugate gradients (the identity preconditioner).

    Attributes:
        krylov: Tolerance and iteration cap of the CG solves (``tol`` defaults to ``1e-10``).
    """

    krylov: KrylovConfig = field(default_factory=_default_krylov)

    def solve(self, op: SymmetricOperator, idx: NDArray[np.int_], rhs: Vector, x0: Vector | None) -> tuple[Vector, int]:
        """Solve the free block ``A[F, F] y = rhs`` by plain CG."""
        return _pcg_block(op, idx, rhs, x0, self.krylov, None)


@dataclass(frozen=True)
class Jacobi:
    """Jacobi-preconditioned CG — runs at the operator's condition number, a bad diagonal scaling removed.

    Attributes:
        krylov: Tolerance and iteration cap of the CG solves (``tol`` defaults to ``1e-10``).
    """

    krylov: KrylovConfig = field(default_factory=_default_krylov)

    def solve(self, op: SymmetricOperator, idx: NDArray[np.int_], rhs: Vector, x0: Vector | None) -> tuple[Vector, int]:
        """Solve the free block ``A[F, F] y = rhs`` by Jacobi-preconditioned CG."""
        return _pcg_block(op, idx, rhs, x0, self.krylov, _jacobi(op, idx))


@dataclass(frozen=True)
class Nystrom:
    """Randomized Nyström-preconditioned CG — for free blocks with a steeply decaying spectrum.

    Attributes:
        krylov: Tolerance and iteration cap of the CG solves (``tol`` defaults to ``1e-10``).
        nystrom: Sketch rank, oversampling, shift and seed of the low-rank
            preconditioner (see :class:`nncg.preconditioners.NystromConfig`).
    """

    krylov: KrylovConfig = field(default_factory=_default_krylov)
    nystrom: NystromConfig = field(default_factory=NystromConfig)

    def solve(self, op: SymmetricOperator, idx: NDArray[np.int_], rhs: Vector, x0: Vector | None) -> tuple[Vector, int]:
        """Solve the free block ``A[F, F] y = rhs`` by Nyström-preconditioned CG (plain CG on an empty block)."""
        precond = _nystrom(op, idx, self.nystrom) if idx.size else None
        return _pcg_block(op, idx, rhs, x0, self.krylov, precond)


@dataclass(frozen=True)
class GlobalNystrom:
    """Nyström-preconditioned CG sketched once on the full operator, then masked per free block.

    :class:`Nystrom` resketches ``A[F, F]`` from scratch on every outer step —
    the `rank + oversample` matrix-free products, a QR, a small Cholesky and an
    SVD, all paid again each time the free set changes. This class instead
    sketches the *full* operator ``A`` once: restricting a rank-``rank``
    factorization to a principal submatrix is exact
    (``(U diag(lam) U^T)[F, F] = U_F diag(lam) U_F^T`` for ``U_F = U[F, :]``),
    so masking rows of the one global basis gives a valid free-block
    preconditioner with no further matrix-free products against ``A`` — only a
    small ``rank x rank`` factorization per free block (see
    :func:`nncg.preconditioners._masked_nystrom`). This amortises well when the
    same operator is solved repeatedly (a parameter sweep, successive warm
    starts) or the active-set loop takes many outer steps; the trade is a
    preconditioner not adapted to each free block's own local spectrum, so it
    can take a few more CG iterations than a freshly-sketched :class:`Nystrom`
    on a small or spectrally unusual free block.

    The global sketch is memoised in a private single slot, keyed on operator
    *identity* so the cache can never carry a stale sketch across operators
    (mirrors :class:`Exact`'s conditioning memo) — excluded from equality/repr
    so this class stays a value.

    Attributes:
        krylov: Tolerance and iteration cap of the CG solves (``tol`` defaults to ``1e-10``).
        nystrom: Sketch rank, oversampling, shift and seed of the global sketch
            (see :class:`nncg.preconditioners.NystromConfig`).
    """

    krylov: KrylovConfig = field(default_factory=_default_krylov)
    nystrom: NystromConfig = field(default_factory=NystromConfig)
    _checked_op: SymmetricOperator | None = field(default=None, compare=False, repr=False)
    _sketch: GlobalNystromSketch | None = field(default=None, compare=False, repr=False)

    def _ensure_sketch(self, op: SymmetricOperator) -> GlobalNystromSketch:
        """Return the memoised global sketch of ``op``, (re)building it on a new operator."""
        sketch = self._sketch
        if self._checked_op is not op or sketch is None:
            sketch = _global_nystrom_sketch(op, self.nystrom)
            object.__setattr__(self, "_sketch", sketch)
            object.__setattr__(self, "_checked_op", op)
        return sketch

    def solve(self, op: SymmetricOperator, idx: NDArray[np.int_], rhs: Vector, x0: Vector | None) -> tuple[Vector, int]:
        """Solve the free block ``A[F, F] y = rhs`` by CG preconditioned from the masked global sketch."""
        precond = _masked_nystrom(self._ensure_sketch(op), idx) if idx.size else None
        return _pcg_block(op, idx, rhs, x0, self.krylov, precond)
