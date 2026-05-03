"""Runtime configuration for agentstate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agentstate.exceptions import AgentStateError
from agentstate.storage.base import BaseCASBackend
from agentstate.storage.memory import InMemoryCAS


DEFAULT_INLINE_THRESHOLD_BYTES = 64 * 1024


@dataclass(frozen=True)
class AgentStateRuntime:
    """Runtime configuration for one AgentState workflow or adapter."""

    backend: BaseCASBackend
    inline_threshold_bytes: int = DEFAULT_INLINE_THRESHOLD_BYTES
    framework: Optional[Any] = None


AgentStateConfig = AgentStateRuntime


_CONFIG = AgentStateRuntime(backend=InMemoryCAS())


def create_runtime(
    *,
    runtime: Optional[AgentStateRuntime] = None,
    backend: Optional[BaseCASBackend] = None,
    inline_threshold_bytes: Optional[int] = None,
    framework: Optional[Any] = None,
) -> AgentStateRuntime:
    """Return a runtime using explicit values over ``runtime`` or global config."""

    base = runtime or _CONFIG
    next_threshold = (
        base.inline_threshold_bytes
        if inline_threshold_bytes is None
        else inline_threshold_bytes
    )
    if next_threshold <= 0:
        raise AgentStateError("inline_threshold_bytes must be a positive integer.")

    return AgentStateRuntime(
        backend=backend or base.backend,
        inline_threshold_bytes=next_threshold,
        framework=base.framework if framework is None else framework,
    )


def configure(
    *,
    backend: Optional[BaseCASBackend] = None,
    inline_threshold_bytes: Optional[int] = None,
    framework: Optional[Any] = None,
) -> AgentStateConfig:
    """Update and return the global fallback runtime.

    Args:
        backend: Content-addressed storage backend for externalized values.
        inline_threshold_bytes: Maximum serialized size for Inline fields.
        framework: Optional explicit framework selection used by later
            detection/adaptation phases.

    Raises:
        AgentStateError: If ``inline_threshold_bytes`` is not positive.
    """

    global _CONFIG

    _CONFIG = create_runtime(
        runtime=_CONFIG,
        backend=backend,
        inline_threshold_bytes=inline_threshold_bytes,
        framework=framework,
    )
    return _CONFIG


def get_config() -> AgentStateConfig:
    """Return the current global fallback runtime."""

    return _CONFIG


def _reset_config_for_tests() -> AgentStateConfig:
    """Reset global configuration to deterministic test defaults."""

    global _CONFIG

    _CONFIG = AgentStateRuntime(backend=InMemoryCAS())
    return _CONFIG
