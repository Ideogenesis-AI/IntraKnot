"""Smoke test: verify that intraknot imports correctly."""

import intraknot


def test_version():
    assert isinstance(intraknot.__version__, str)
