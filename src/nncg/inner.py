"""Inner solvers: one free-block system ``A[F, F] y = rhs`` per active-set step.

Each concrete inner solver provides ``solve(op, idx, rhs, x0) -> (y, iters)``,
solving the free-block system ``A[F, F] y = rhs`` (and so satisfies the
:class:`nncg.solver.InnerSolver` interface). This is the only module that knows
about preconditioning: the built-in solvers are the identity/Jacobi/Nyström-
preconditioned CG variants (:class:`CG`, :class:`Jacobi`, :class:`Nystrom`) and
the direct :class:`Exact`; further inner solvers — e.g. Clarabel- or
KKT-equation-based — live in Jebel-Quant/mean_variance_solvers and satisfy the
same structural interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import cast

import numpy as np
from cvx.linalg import SymmetricOperator, Vector
from numpy.typing import NDArray

from .krylov import KrylovConfig, MatVec, Preconditioner, pcg

_RCOND_MIN = 1e-12  # matches cvx-linalg's DEFAULT_COND_THRESHOLD of 1e12


@dataclass(frozen=True)
class NystromConfig:
    """Tuning knobs for the Nyström preconditioner of :class:`Nystrom`.

    Attributes:
        rank: Target sketch rank — the number of leading eigenpairs captured,
            clamped to the free-block dimension.
        oversample: Extra sketch columns drawn for accuracy before truncating
            back to ``rank`` (the standard randomized-SVD oversampling).
        shift: Explicit scalar tail eigenvalue, or ``None`` for the default
            (the largest eigenvalue the sketch does not capture).
        seed: Seed for the Gaussian test matrix; fixed by default so a solve is
            reproducible. ``None`` draws a fresh one.

    Raises:
        ValueError: When ``rank < 1``.
    """

    rank: int = 10
    oversample: int = 10
    shift: float | None = None
    seed: int | None = 0

    def __post_init__(self) -> None:
        """Validate that the sketch rank is a positive integer."""
        if self.rank < 1:
            msg = f"NystromConfig.rank must be a positive integer; got {self.rank}"
            raise ValueError(msg)


#: Shared default so ``NystromConfig()`` is not called in argument defaults (ruff B008).
_DEFAULT_NYSTROM = NystromConfig()


def _free_matvec(op: SymmetricOperator, idx: NDArray[np.int_]) -> MatVec:
    """Return the free-block action ``v -> A[F, F] v`` of the operator.

    The free-set restriction is hoisted out of the inner loop: the pre-sliced
    free-block operator is built once here and the returned callable is its
    plain ``matvec``. Restricting per CG iteration instead re-gathers the
    operator's storage on every call, an order of magnitude more wall clock.

    Args:
        op: The symmetric operator ``A``.
        idx: Integer positions of the free set ``F``.

    Returns:
        A callable computing ``A[F, F] @ v``; the reduced matrix is never
        materialised.
    """
    return cast(MatVec, op.restricted(idx).matvec)


def _jacobi(op: SymmetricOperator, idx: NDArray[np.int_] | None = None) -> Preconditioner:
    """Return the Jacobi preconditioner ``r -> (1 / diag(A))[F] * r``.

    With ``idx`` given it is sliced to the free block ``A[F, F]``; ``idx=None``
    gives the whole operator. The diagonal is read off ``op.diag`` (matrix never
    materialised). A symmetric positive definite operator has a strictly positive
    diagonal, so a non-positive or non-finite entry means ``A`` is not SPD there;
    that is reported eagerly rather than propagated as an ``inf`` into the CG loop.

    Args:
        op: The SPD operator ``A``.
        idx: Integer positions of the free set ``F``, or ``None`` for all of ``A``.

    Returns:
        The elementwise preconditioner (sliced to the free set when ``idx`` given).

    Raises:
        ValueError: When a diagonal entry is non-positive or non-finite.
        NotImplementedError: When the backend does not expose ``diag``.
    """
    diag = np.asarray(op.diag, dtype=np.float64)
    bad = np.flatnonzero(~(diag > 0.0) | ~np.isfinite(diag))
    if bad.size:
        i = int(bad[0])
        msg = f"operator diagonal is not strictly positive at index {i} (diag={diag[i]:.2e}); A is not SPD"
        raise ValueError(msg)
    dinv = 1.0 / diag
    if idx is not None:
        dinv = dinv[idx]
    return lambda r: dinv * r


def _nystrom_sketch(
    matvec: MatVec, n: int, sketch: int, seed: int | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the randomized Nyström eigendecomposition of the operator.

    Draws an ``n x sketch`` orthonormal test matrix, forms ``Y = A Omega`` with
    ``sketch`` matrix-free products, and applies the Frangella-Tropp-Udell
    stabilised sketch (Alg. 2.1): a nugget shift lifts ``Y`` off the range
    boundary so the small Cholesky is well conditioned, then a thin SVD yields
    the orthonormal basis and the nugget-corrected, clipped eigenvalues.

    Args:
        matvec: The matrix-free action ``v -> A v`` (already free-block sliced).
        n: Dimension of the (free-block) operator.
        sketch: Number of test columns (``rank + oversample``, clamped to ``n``).
        seed: Seed for the Gaussian test matrix (``None`` draws a fresh one).

    Returns:
        ``(u_full, lam_full)``: the orthonormal basis and eigenvalues in
        descending order, before truncation to the requested rank.
    """
    rng = np.random.default_rng(seed)
    omega = np.linalg.qr(rng.standard_normal((n, sketch)))[0]  # n x sketch, orthonormal
    y = np.column_stack([matvec(omega[:, j]) for j in range(sketch)])  # A_F @ Omega

    # Stabilising shift (Frangella-Tropp-Udell Alg. 2.1): lift Y off the range
    # boundary so the small Cholesky is well conditioned, then subtract it back.
    nu = np.sqrt(n) * np.finfo(np.float64).eps * float(np.linalg.norm(y, ord=2))
    y_nu = y + nu * omega
    chol = np.linalg.cholesky(omega.T @ y_nu)  # lower, chol @ chol.T = Omega^T Y_nu
    b = np.linalg.solve(chol, y_nu.T).T  # B = Y_nu chol^{-T}, so B B^T = Y_nu (Omega^T Y_nu)^{-1} Y_nu^T
    u_full, sv, _ = np.linalg.svd(b, full_matrices=False)
    lam_full = np.maximum(sv**2 - nu, 0.0)  # eigenvalues of the Nystrom approximation
    return u_full, lam_full


def _check_captured(lam: NDArray[np.float64], rank: int) -> None:
    """Validate that the rank-``rank`` sketch captured a genuine eigenspace.

    The smallest captured eigenvalue must be a real positive eigenvalue, not
    floating-point noise from a rank the block does not possess.

    Args:
        lam: The captured eigenvalues in descending order.
        rank: The requested sketch rank (for the error message).

    Raises:
        ValueError: When the smallest captured eigenvalue is non-positive or
            negligible relative to the largest — ``rank`` exceeds the numerical
            rank of the block, so reduce it.
    """
    if float(lam[0]) <= 0.0 or float(lam[-1]) <= 1e-12 * float(lam[0]):
        ratio = float(lam[-1]) / float(lam[0]) if float(lam[0]) > 0.0 else 0.0
        msg = (
            f"the rank-{rank} Nystrom sketch captured a negligible eigenvalue "
            f"(lam_min/lam_max={ratio:.2e}); rank exceeds the numerical rank of A — reduce it"
        )
        raise ValueError(msg)


def _nystrom_shift(shift: float | None, lam: NDArray[np.float64], lam_full: NDArray[np.float64], rank: int) -> float:
    """Choose the scalar deflation shift ``mu`` for the uncaptured spectral tail.

    Uses an explicit ``shift`` when given; otherwise the largest *uncaptured*
    eigenvalue (deflation), falling back to the smallest captured one when the
    sketch spanned the whole spectrum (``oversample=0``).

    Args:
        shift: Explicit shift, or ``None`` for the default deflation choice.
        lam: The captured eigenvalues in descending order.
        lam_full: All sketched eigenvalues (captured plus tail) in descending order.
        rank: The requested sketch rank.

    Returns:
        The positive scalar shift ``mu``.

    Raises:
        ValueError: When the resolved shift is not positive.
    """
    if shift is not None:
        mu = float(shift)
    elif lam_full.size > rank and float(lam_full[rank]) > 1e-12 * float(lam[0]):
        mu = float(lam_full[rank])  # largest uncaptured eigenvalue: the deflation shift
    else:
        mu = float(lam[-1])  # sketch captured the whole spectrum: fall back to smallest captured
    if mu <= 0.0:
        msg = f"shift must be positive; got {mu:.2e}"
        raise ValueError(msg)
    return mu


def _nystrom(
    op: SymmetricOperator, idx: NDArray[np.int_] | None = None, config: NystromConfig = _DEFAULT_NYSTROM
) -> Preconditioner:
    """Return a randomized Nyström preconditioner for ``A`` (or its free block ``A[F, F]``).

    A rank-``rank`` randomized Nyström sketch approximates the block as
    ``A_F ~ U diag(lam) U^T`` (Frangella, Tropp & Udell, 2023). Treating the
    uncaptured tail as a single scalar ``shift`` gives the SPD preconditioner
    ``M = U diag(lam) U^T + shift (I - U U^T)``, applied by the Woodbury formula
    ``M^{-1} r = (1/shift) r + U ((1/lam - 1/shift) * (U^T r))`` in ``O(|F| rank)``
    per call — ``U``, ``lam``, ``shift`` are captured once at build time (the
    ``rank + oversample`` matrix-free products and a small dense factorisation).
    The default ``shift`` is the largest *uncaptured* eigenvalue (deflation),
    falling back to the smallest captured one when ``oversample=0``.

    Args:
        op: The SPD operator ``A``.
        idx: Integer positions of the free set ``F``.
        config: Sketch rank, oversampling, shift and seed.

    Returns:
        A callable applying ``r -> M^{-1} r`` on the free block.

    Raises:
        ValueError: When the smallest captured eigenvalue is negligible relative
            to the largest (``config.rank`` exceeds the numerical rank of the
            block — reduce it), or when an explicit ``config.shift`` is not
            positive.
    """
    rank, oversample, shift, seed = config.rank, config.oversample, config.shift, config.seed
    matvec: MatVec
    if idx is None:
        matvec, n = op.matvec, op.n
    else:
        matvec, n = _free_matvec(op, idx), int(idx.size)
    rank = min(rank, n)
    sketch = min(rank + max(oversample, 0), n)

    u_full, lam_full = _nystrom_sketch(matvec, n, sketch, seed)
    u = u_full[:, :rank]
    lam = lam_full[:rank]
    _check_captured(lam, rank)
    mu = _nystrom_shift(shift, lam, lam_full, rank)

    inv_mu = 1.0 / mu
    coef = 1.0 / lam - inv_mu  # low-rank correction weights; <= 0 for the default shift

    def apply(r: Vector) -> Vector:
        """Apply ``M^{-1}`` via the captured scalar shift plus low-rank correction."""
        z: Vector = inv_mu * r + u @ (coef * (u.T @ r))
        return z

    return apply


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
            rcond = op.rcond_free(idx)
            if rcond < _RCOND_MIN:
                msg = f"free block of size {idx.size} is numerically singular (rcond={rcond:.2e})"
                raise ValueError(msg)
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
            preconditioner (see :class:`NystromConfig`).
    """

    krylov: KrylovConfig = field(default_factory=_default_krylov)
    nystrom: NystromConfig = field(default_factory=NystromConfig)

    def solve(self, op: SymmetricOperator, idx: NDArray[np.int_], rhs: Vector, x0: Vector | None) -> tuple[Vector, int]:
        """Solve the free block ``A[F, F] y = rhs`` by Nyström-preconditioned CG (plain CG on an empty block)."""
        precond = _nystrom(op, idx, self.nystrom) if idx.size else None
        return _pcg_block(op, idx, rhs, x0, self.krylov, precond)
