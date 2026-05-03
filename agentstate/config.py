"""Global configuration for agentstate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agentstate.exceptions import AgentStateError
from agentstate.storage.base import BaseCASBackend
from agentstate.storage.memory import InMemoryCAS


DEFAULT_INLINE_THRESHOLD_BYTES = 64 * 1024


@dataclass(frozen=True)
class AgentStateConfig:
    """Runtime configuration shared by descriptors and adapters."""

    backend: BaseCASBackend
    inline_threshold_bytes: int = DEFAULT_INLINE_THRESHOLD_BYTES
    framework: Optional[Any] = None


_CONFIG = AgentStateConfig(backend=InMemoryCAS())


def configure(
    *,
    backend: Optional[BaseCASBackend] = None,
    inline_threshold_bytes: Optional[int] = None,
    framework: Optional[Any] = None,
) -> AgentStateConfig:
    """Update and return the global agentstate configuration.

    Args:
        backend: Content-addressed storage backend for externalized values.
        inline_threshold_bytes: Maximum serialized size for Inline fields.
        framework: Optional explicit framework selection used by later
            detection/adaptation phases.

    Raises:
        AgentStateError: If ``inline_threshold_bytes`` is not positive.
    """

    global _CONFIG

    next_threshold = (
        _CONFIG.inline_threshold_bytes
        if inline_threshold_bytes is None
        else inline_threshold_bytes
    )
    if next_threshold <= 0:
        raise AgentStateError("inline_threshold_bytes must be a positive integer.")

    _CONFIG = AgentStateConfig(
        backend=backend or _CONFIG.backend,
        inline_threshold_bytes=next_threshold,
        framework=_CONFIG.framework if framework is None else framework,
    )
    return _CONFIG


def get_config() -> AgentStateConfig:
    """Return the current global agentstate configuration."""

    return _CONFIG


def _reset_config_for_tests() -> AgentStateConfig:
    """Reset global configuration to deterministic test defaults."""

    global _CONFIG

    _CONFIG = AgentStateConfig(backend=InMemoryCAS())
    return _CONFIG
