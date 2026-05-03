"""Framework adapters for AgentState."""

from __future__ import annotations

from typing import Any, Optional, Type

from agentstate.adapters.autogen import AutoGenAdapter
from agentstate.adapters.base import BaseFrameworkAdapter
from agentstate.adapters.langgraph import LangGraphAdapter
from agentstate.adapters.llamaindex import LlamaIndexAdapter
from agentstate.config import AgentStateRuntime
from agentstate.core.state import AgentState
from agentstate.detection.framework import Framework, detect_active_framework
from agentstate.storage.base import BaseCASBackend


def get_adapter(
    framework: Framework,
    state_cls: Optional[Type[AgentState]] = None,
    *,
    runtime: Optional[AgentStateRuntime] = None,
    backend: Optional[BaseCASBackend] = None,
    inline_threshold_bytes: Optional[int] = None,
) -> BaseFrameworkAdapter:
    """Return the adapter for ``framework``."""

    if framework is Framework.LANGGRAPH:
        return LangGraphAdapter(
            state_cls,
            runtime=runtime,
            backend=backend,
            inline_threshold_bytes=inline_threshold_bytes,
        )
    if framework is Framework.LLAMAINDEX:
        return LlamaIndexAdapter(
            state_cls,
            runtime=runtime,
            backend=backend,
            inline_threshold_bytes=inline_threshold_bytes,
        )
    if framework is Framework.AUTOGEN:
        return AutoGenAdapter(
            state_cls,
            runtime=runtime,
            backend=backend,
            inline_threshold_bytes=inline_threshold_bytes,
        )
    raise ValueError(f"Unsupported framework: {framework!r}")


def auto_adapt(
    state_cls: Type[AgentState],
    explicit: Optional[Framework] = None,
    *,
    runtime: Optional[AgentStateRuntime] = None,
    backend: Optional[BaseCASBackend] = None,
    inline_threshold_bytes: Optional[int] = None,
) -> Any:
    """Auto-detect the active framework and adapt ``state_cls`` for it."""

    framework = detect_active_framework(explicit)
    return get_adapter(
        framework,
        state_cls,
        runtime=runtime,
        backend=backend,
        inline_threshold_bytes=inline_threshold_bytes,
    ).schema()


__all__ = [
    "AutoGenAdapter",
    "BaseFrameworkAdapter",
    "LangGraphAdapter",
    "LlamaIndexAdapter",
    "auto_adapt",
    "get_adapter",
]
