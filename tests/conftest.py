"""Shared test fixtures."""

from __future__ import annotations

from typing import Generator

import pytest

from agentref.config import _reset_config_for_tests


@pytest.fixture(autouse=True)
def reset_agentref_config() -> Generator[None, None, None]:
    """Reset global configuration before and after each test."""

    _reset_config_for_tests()
    yield
    _reset_config_for_tests()
