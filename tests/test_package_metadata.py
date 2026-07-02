"""Package metadata sanity checks."""

import importlib
import importlib.metadata

import pytest

import nncg


def test_version_present() -> None:
    """The package exposes a version string."""
    assert isinstance(nncg.__version__, str)
    assert nncg.__version__


def test_version_fallback_without_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without installed package metadata the version falls back to 0.0.0."""

    def raise_not_found(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)
    try:
        importlib.reload(nncg)
        assert nncg.__version__ == "0.0.0"
    finally:
        monkeypatch.undo()
        importlib.reload(nncg)  # restore the real version for the other tests


def test_public_api_exported() -> None:
    """Everything in __all__ is importable from the package root."""
    for name in nncg.__all__:
        assert hasattr(nncg, name)
