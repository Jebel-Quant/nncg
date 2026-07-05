"""Preconditioners for the preconditioned CG inner solver.

The matrix-free PCG in :mod:`nncg.krylov` applies its preconditioner only
through the action ``z = M^{-1} r``, so every preconditioner here is a plain
callable ``r -> M^{-1} r`` (the :data:`nncg.krylov.Preconditioner` type). That
one abstraction covers both flavours this package uses:

* :func:`diagonal` / :func:`jacobi` — the diagonal preconditioner
  ``M^{-1} = diag(dinv)``, read off the operator's diagonal without forming the
  matrix. This is what the active-set loop in :mod:`nncg.solver` uses for
  ``inner="pcg"``, restricted to each free set as ``diagonal(dinv[F])``.
* :func:`nystrom` — the randomized Nyström preconditioner: a rank-``r`` sketch
  ``A ~ U diag(lam) U^T`` built from ``r`` matrix-free products, applied as a
  scalar shift plus a low-rank Woodbury correction in ``O(n r)`` per solve. Use
  it on a *fixed* SPD system whose spectrum decays — the sketch pays for itself
  only when the low-rank capture collapses the condition number.
"""

from __future__ import annotations

import numpy as np
from cvx.linalg import SymmetricOperator, Vector

from .krylov import MatVec, Preconditioner


def inverse_diagonal(op: SymmetricOperator) -> Vector:
    """Return ``dinv = 1 / diag(A)``, the raw Jacobi scaling of an operator.

    This is the vector underlying :func:`jacobi`; call it directly when you
    need to slice the diagonal (the active-set loop restricts it to a free set
    as ``dinv[F]``). The diagonal is read off ``op.diag``, so the matrix is
    never materialised.

    A symmetric positive definite operator has a strictly positive diagonal
    (``e_i^T A e_i > 0``), so a non-positive or non-finite entry means ``A`` is
    not SPD on the diagonal and the reciprocal would be meaningless; this is
    reported eagerly rather than propagated as an ``inf`` into the CG loop.

    Args:
        op: The SPD operator ``A`` (a :class:`cvx.linalg.SymmetricOperator`).

    Returns:
        The elementwise inverse of the operator's diagonal, of length ``op.n``.

    Raises:
        ValueError: When any diagonal entry is non-positive or non-finite —
            ``A`` is then not positive definite; add a ridge.
        NotImplementedError: When the backend does not expose ``diag``
            (propagated from :mod:`cvx.linalg`).
    """
    diag = np.asarray(op.diag, dtype=np.float64)
    bad = np.flatnonzero(~(diag > 0.0) | ~np.isfinite(diag))
    if bad.size:
        i = int(bad[0])
        msg = f"operator diagonal is not strictly positive at index {i} (diag={diag[i]:.2e}); A is not SPD"
        raise ValueError(msg)
    return 1.0 / diag


def identity() -> Preconditioner:
    """Return the identity preconditioner ``M^{-1} = I``.

    Passing this to :func:`nncg.krylov.pcg` reproduces plain CG iterate for
    iterate — a useful control when isolating the effect of a preconditioner in
    the numerical study.

    Returns:
        A callable returning its argument unchanged.
    """
    return lambda r: r


def diagonal(dinv: Vector) -> Preconditioner:
    """Return the diagonal preconditioner ``M^{-1} = diag(dinv)``.

    The returned callable applies ``r -> dinv * r``. Build ``dinv`` with
    :func:`inverse_diagonal` for the Jacobi preconditioner, or pass any
    positive scaling of your own.

    Args:
        dinv: The diagonal of ``M^{-1}`` (elementwise inverse of ``diag(M)``).

    Returns:
        A callable applying the elementwise product ``dinv * r``.
    """
    return lambda r: dinv * r


def jacobi(op: SymmetricOperator) -> Preconditioner:
    """Return the Jacobi preconditioner ``M^{-1} = diag(1 / diag(A))``.

    Convenience wrapper over ``diagonal(inverse_diagonal(op))`` for a standalone
    :func:`nncg.krylov.pcg` solve. The active-set loop instead slices the raw
    :func:`inverse_diagonal` per free set, so it does not go through here.

    Args:
        op: The SPD operator ``A`` (a :class:`cvx.linalg.SymmetricOperator`).

    Returns:
        The diagonal preconditioner built from the operator's diagonal.

    Raises:
        ValueError: When the diagonal is not strictly positive (see
            :func:`inverse_diagonal`).
        NotImplementedError: When the backend does not expose ``diag``.
    """
    return diagonal(inverse_diagonal(op))


def nystrom(
    matvec: MatVec,
    n: int,
    rank: int,
    oversample: int = 10,
    shift: float | None = None,
    seed: int | None = None,
) -> Preconditioner:
    """Return a randomized Nyström preconditioner for an SPD operator.

    A rank-``rank`` randomized Nyström sketch approximates the operator as
    ``A ~ U diag(lam) U^T`` (Frangella, Tropp & Udell, *Randomized Nyström
    Preconditioning*, 2023). Treating the uncaptured tail of the spectrum as a
    single scalar ``shift`` gives the SPD preconditioner

        ``M = U diag(lam) U^T + shift (I - U U^T)``,

    whose inverse is applied by the Woodbury / low-rank formula

        ``M^{-1} r = (1/shift) r + U ((1/lam - 1/shift) * (U^T r))``

    in ``O(n * rank)`` per call — the ``U``, ``lam`` and ``shift`` are captured
    once at build time (the ``rank + oversample`` matrix-free products and a
    small dense factorisation), not recomputed per application. The deflated
    operator ``M^{-1} A`` then has condition number about ``shift / lam_min``,
    so the sketch helps when the spectrum decays fast enough for a modest
    ``rank`` to lift the leading eigenvalues off a well-conditioned tail.

    Because the sketch is tied to this particular ``matvec``, the preconditioner
    is built for one fixed SPD system; the active-set loop's free blocks each
    differ, so wiring it there would rebuild the sketch every outer step.

    Unlike :func:`jacobi` / :func:`inverse_diagonal`, this builder takes a bare
    ``matvec`` (with an explicit ``n``) rather than a
    :class:`cvx.linalg.SymmetricOperator`. The rule across this module is that
    each builder asks only for the capability it needs: the diagonal
    preconditioners need ``op.diag`` — not recoverable from products without
    ``n`` probe mat-vecs — so they require the operator, whereas Nyström is
    intrinsically matrix-free (a handful of products *is* the algorithm) and
    needs nothing more. Keeping it mat-vec based also lets the ``inner="nystrom"``
    inner solver precondition each free block ``A[F, F]`` directly, on backends
    that expose only the free-block action and no restricted operator.

    Args:
        matvec: The action ``v -> A v`` of the SPD operator to precondition.
        n: Dimension of the operator.
        rank: Target sketch rank (number of leading eigenpairs captured).
            Clamped to ``n``.
        oversample: Extra sketch columns drawn for accuracy before truncating
            back to ``rank`` (the standard randomized-SVD oversampling).
        shift: The scalar tail eigenvalue that stands in for everything the
            rank-``rank`` block does not capture. Defaults to the largest
            *uncaptured* eigenvalue ``lam[rank]`` (read from the oversampled
            sketch) — the deflation choice that leaves ``M^{-1} A`` conditioned
            by the tail alone. Falls back to the smallest captured eigenvalue
            when ``oversample=0`` leaves no tail estimate.
        seed: Seed for the Gaussian test matrix; ``None`` draws a fresh one.

    Returns:
        A callable applying ``r -> M^{-1} r``.

    Raises:
        ValueError: When ``rank < 1``, or when the smallest captured
            eigenvalue is negligible relative to the largest (``rank`` exceeds
            the numerical rank of the operator — reduce it), or when an
            explicit ``shift`` is not positive.
    """
    if rank < 1:
        msg = f"rank must be a positive integer; got {rank}"
        raise ValueError(msg)
    rank = min(rank, n)
    sketch = min(rank + max(oversample, 0), n)

    rng = np.random.default_rng(seed)
    omega = np.linalg.qr(rng.standard_normal((n, sketch)))[0]  # n x sketch, orthonormal
    y = np.column_stack([matvec(omega[:, j]) for j in range(sketch)])  # A @ Omega

    # Stabilising shift (Frangella-Tropp-Udell Alg. 2.1): lift Y off the range
    # boundary so the small Cholesky is well conditioned, then subtract it back.
    nu = np.sqrt(n) * np.finfo(np.float64).eps * float(np.linalg.norm(y, ord=2))
    y_nu = y + nu * omega
    chol = np.linalg.cholesky(omega.T @ y_nu)  # lower, chol @ chol.T = Omega^T Y_nu
    b = np.linalg.solve(chol, y_nu.T).T  # B = Y_nu chol^{-T}, so B B^T = Y_nu (Omega^T Y_nu)^{-1} Y_nu^T
    u_full, sv, _ = np.linalg.svd(b, full_matrices=False)
    lam_full = np.maximum(sv**2 - nu, 0.0)  # eigenvalues of the Nystrom approximation

    u = u_full[:, :rank]
    lam = lam_full[:rank]
    # The smallest captured eigenvalue is the default shift, so it must be a
    # genuine positive eigenvalue, not floating-point noise from a rank the
    # operator does not have (the null directions leave lam ~ eps^2 * lam[0]).
    if float(lam[0]) <= 0.0 or float(lam[-1]) <= 1e-12 * float(lam[0]):
        ratio = float(lam[-1]) / float(lam[0]) if float(lam[0]) > 0.0 else 0.0
        msg = (
            f"the rank-{rank} Nystrom sketch captured a negligible eigenvalue "
            f"(lam_min/lam_max={ratio:.2e}); rank exceeds the numerical rank of A — reduce it"
        )
        raise ValueError(msg)
    if shift is not None:
        mu = float(shift)
    elif lam_full.size > rank and float(lam_full[rank]) > 1e-12 * float(lam[0]):
        mu = float(lam_full[rank])  # largest uncaptured eigenvalue: the deflation shift
    else:
        mu = float(lam[-1])  # sketch captured the whole spectrum: fall back to smallest captured
    if mu <= 0.0:
        msg = f"shift must be positive; got {mu:.2e}"
        raise ValueError(msg)

    inv_mu = 1.0 / mu
    coef = 1.0 / lam - inv_mu  # low-rank correction weights; <= 0 for the default shift

    def apply(r: Vector) -> Vector:
        """Apply ``M^{-1}`` via the captured scalar shift plus low-rank correction."""
        z: Vector = inv_mu * r + u @ (coef * (u.T @ r))
        return z

    return apply
