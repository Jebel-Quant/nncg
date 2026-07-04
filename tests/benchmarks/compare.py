"""Iteration/time/accuracy comparison of ``nncg`` against the baseline solvers.

A runnable harness (``uv run python -m tests.benchmarks.compare``) that solves
the planted-optimum families with :func:`nncg.solver.solve_nnqp` /
:func:`nncg.solver.solve_nnqp_eq` and each alternative in
:mod:`tests.baselines`, then tabulates iterations, wall-clock time, distance to
the planted optimum, and the KKT / feasibility residual. It is a study
harness, not a test — the numbers are informative, not asserted.

The bound-only sweep pits ``nncg`` against OSQP, Clarabel and Lawson-Hanson;
the ``p = 1`` simplex sweep adds Duchi, which is defined only there. OSQP and
Clarabel are optional — rows are emitted only when they import.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from cvx.linalg import DenseOperator, SymmetricOperator, Vector

from nncg import kkt_violation, solve_nnqp, solve_nnqp_eq
from tests import baselines as bl
from tests.problems import make_problem, make_simplex_problem


@dataclass(frozen=True)
class Row:
    """One line of the comparison table.

    Attributes:
        problem: Label of the problem instance.
        solver: Name of the solver.
        iters: Native iteration count reported by the solver.
        time_s: Wall-clock solve time in seconds.
        err: Distance ``||x - x_star||_inf`` to the planted optimum.
        residual: KKT violation (bound-only) or ``max(KKT-ish, |1^T x - beta|)``.
        status: Solver status string.
    """

    problem: str
    solver: str
    iters: int
    time_s: float
    err: float
    residual: float
    status: str


def _bound_rows(op: SymmetricOperator, b: Vector, x_star: Vector, label: str) -> list[Row]:
    """Solve one bound-only instance with every applicable solver.

    Args:
        op: The SPD operator ``A``.
        b: The linear term ``b``.
        x_star: The planted optimum.
        label: Problem label for the table.

    Returns:
        One :class:`Row` per solver that is available.
    """
    scale = 1.0 + float(np.linalg.norm(b))
    rows: list[Row] = []

    t0 = perf_counter()
    r = solve_nnqp(op, b)
    t_nncg = perf_counter() - t0
    rows.append(
        Row(
            label,
            "nncg (cg)",
            r.inner,
            t_nncg,
            float(np.max(np.abs(r.x - x_star))),
            kkt_violation(op, b, r.x) / scale,
            "solved" if r.converged else "capped",
        )
    )
    for name, fn in [*_optional_bound_solvers(), ("lawson-hanson (cg)", bl.solve_lawson_hanson)]:
        res = fn(op, b)
        rows.append(
            Row(
                label,
                name,
                res.iters,
                res.time_s,
                float(np.max(np.abs(res.x - x_star))),
                kkt_violation(op, b, res.x) / scale,
                res.status,
            )
        )
    return rows


def _simplex_rows(op: SymmetricOperator, b: Vector, beta: float, x_star: Vector, label: str) -> list[Row]:
    """Solve one ``p = 1`` simplex instance with every applicable solver.

    Args:
        op: The SPD operator ``A``.
        b: The linear term ``b``.
        beta: The simplex scale ``1^T x = beta``.
        x_star: The planted optimum.
        label: Problem label for the table.

    Returns:
        One :class:`Row` per solver, ``nncg_eq`` and Duchi always, OSQP and
        Clarabel when available.
    """
    b_eq, c_eq = bl.ones_row(op.n), np.array([beta])

    def feas(x: Vector) -> float:
        """Feasibility residual: distance from the simplex plus KKT slack."""
        return float(max(abs(x.sum() - beta), np.max(-x, initial=0.0)))

    rows: list[Row] = []
    t0 = perf_counter()
    r = solve_nnqp_eq(op, b, b_eq, c_eq)
    t_nncg = perf_counter() - t0
    rows.append(
        Row(
            label,
            "nncg_eq (cg)",
            r.inner,
            t_nncg,
            float(np.max(np.abs(r.x - x_star))),
            feas(r.x),
            "solved" if r.converged else "capped",
        )
    )

    for name, fn in _optional_eq_solvers():
        res = fn(op, b, b_eq=b_eq, c_eq=c_eq)
        rows.append(
            Row(label, name, res.iters, res.time_s, float(np.max(np.abs(res.x - x_star))), feas(res.x), res.status)
        )

    d = bl.solve_duchi(op, b, beta=beta)
    rows.append(
        Row(label, "duchi (fista)", d.iters, d.time_s, float(np.max(np.abs(d.x - x_star))), feas(d.x), d.status)
    )
    return rows


def _optional_bound_solvers() -> list[tuple[str, object]]:
    """Return the (name, fn) pairs for OSQP/Clarabel that import successfully."""
    out: list[tuple[str, object]] = []
    for name, fn in [("osqp", bl.solve_osqp), ("clarabel", bl.solve_clarabel)]:
        try:
            __import__(name)
        except ImportError:
            continue
        out.append((name, fn))
    return out


def _optional_eq_solvers() -> list[tuple[str, object]]:
    """Return the (name, fn) pairs for OSQP/Clarabel usable with an equality."""
    return _optional_bound_solvers()


def run_comparison(n: int = 200, kappa: float = 1e3) -> list[Row]:
    """Run the full comparison and return the table rows.

    Args:
        n: Problem dimension of every instance.
        kappa: Spectral condition number of the bound-only instance (the
            simplex instance uses ``kappa / 10`` to keep the first-order
            method's iteration count readable).

    Returns:
        The list of :class:`Row`, bound-only instance first.
    """
    rows: list[Row] = []
    a, b, x_star, _ = make_problem(n, kappa, seed=0)
    rows += _bound_rows(DenseOperator(a), b, x_star, f"bound n={n} k={kappa:.0e}")

    a2, b2, beta, x2, _, _ = make_simplex_problem(n, kappa / 10, beta=1.0, seed=0)
    rows += _simplex_rows(DenseOperator(a2), b2, beta, x2, f"simplex n={n} k={kappa / 10:.0e}")
    return rows


def format_table(rows: list[Row]) -> str:
    """Render the comparison rows as a fixed-width text table.

    Args:
        rows: The rows to render.

    Returns:
        A multi-line string with one header and one line per row, blank-line
        separated between problem groups.
    """
    header = (
        f"{'problem':22s} {'solver':20s} {'iters':>7s} {'time[ms]':>9s} {'||x-x*||':>10s} {'residual':>10s}  status"
    )
    lines = [header, "-" * len(header)]
    prev = rows[0].problem if rows else ""
    for row in rows:
        if row.problem != prev:
            lines.append("")
            prev = row.problem
        cells = f"{row.problem:22s} {row.solver:20s} {row.iters:7d} {row.time_s * 1e3:9.2f}"
        cells += f" {row.err:10.2e} {row.residual:10.2e}  {row.status}"
        lines.append(cells)
    return "\n".join(lines)


def main() -> None:
    """Run the comparison and print the table to stdout."""
    rows = run_comparison()
    print(format_table(rows))


if __name__ == "__main__":
    main()
