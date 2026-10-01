"""The README's ``pycon`` transcripts, executed as one doctest session.

The README examples are written as ```` ```pycon ```` fences, which the template's
README check does not run. This test collects every such fence in order and runs
them as a single doctest, because later fences reuse names (``A``, ``b``, ``op``)
bound by earlier ones. Any example whose output or assertion drifts fails the suite.
"""

from __future__ import annotations

import doctest
import re
from pathlib import Path

_README = Path(__file__).absolute().parent.parent / "README.md"
_PYCON = re.compile(r"^```pycon\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _run_transcript(source: str) -> doctest.TestResults:
    """Run ``source`` as one doctest session with ``ELLIPSIS``.

    Args:
        source: Interactive-session text in ``>>>`` notation.

    Returns:
        The runner's failed/attempted counts.
    """
    test = doctest.DocTestParser().get_doctest(source, {}, "README.md", str(_README), 0)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    runner.run(test)
    return runner.summarize(verbose=False)


def test_readme_pycon_examples_pass() -> None:
    """Every ``pycon`` fence in the README runs and matches its shown output."""
    fences = _PYCON.findall(_README.read_text(encoding="utf-8"))
    assert fences, "README.md has no ```pycon fences to check"
    results = _run_transcript("\n".join(fences))
    assert results.attempted > 0
    assert results.failed == 0, f"{results.failed} README example(s) failed; see output above"


def test_runner_reports_a_wrong_expected_output() -> None:
    """A transcript whose shown output is wrong is counted as a failure."""
    results = _run_transcript(">>> 1 + 1\n3\n")
    assert results.failed == 1
