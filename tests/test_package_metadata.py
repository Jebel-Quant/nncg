"""Package metadata sanity checks."""

import nncg


def test_version_present() -> None:
    """The package exposes a version string."""
    assert isinstance(nncg.__version__, str)
    assert nncg.__version__


def test_public_api_exported() -> None:
    """Everything in __all__ is importable from the package root."""
    for name in nncg.__all__:
        assert hasattr(nncg, name)
