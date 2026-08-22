"""Smoke tests for package metadata."""

from vectorchain import __version__


def test_package_exposes_a_development_version() -> None:
    """The installed package must expose its PEP 440 version."""
    assert __version__ == "0.1.0.dev0"
