"""Benchmark harnesses comparing ``nncg`` against the alternative solvers.

Kept out of the normal ``make test`` run (pytest ignores ``tests/benchmarks``)
because these are timing/comparison harnesses, not correctness assertions.
Run the comparison table with ``uv run python -m tests.benchmarks.compare``.
"""
