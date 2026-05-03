"""Framework adapters for AgentState."""

from __future__ import annotations

from typing import Any, Optional, Type

from agentstate.adapters.autogen import AutoGenAdapter
from agentstate.adapters.base import BaseFrameworkAdapter
from agentstate.adapters.langgraph import LangGraphAdapter
from agentstate.adapters.llamaindex import LlamaIndexAdapter
from agentstate.core.state import AgentState
from agentstate.detection.framework import Framework, detect_active_framework


def get_adapter(
    framework: Framework,
    state_cls: Optional[Type[AgentState]] = None,
) -> BaseFrameworkAdapter:
    """Return the adapter for ``framework``."""

    if framework is Framework.LANGGRAPH:
        return LangGraphAdapter(state_cls)
    if framework is Framework.LLAMAINDEX:
        return LlamaIndexAdapter(state_cls)
    if framework is Framework.AUTOGEN:
        return AutoGenAdapter(state_cls)
    raise ValueError(f"Unsupported framework: {framework!r}")


def auto_adapt(state_cls: Type[AgentState], explicit: Optional[Framework] = None) -> Any:
    """Auto-detect the active framework and adapt ``state_cls`` for it."""

    framework = detect_active_framework(explicit)
    return get_adapter(framework, state_cls).schema()


__all__ = [
    "AutoGenAdapter",
    "BaseFrameworkAdapter",
    "LangGraphAdapter",
    "LlamaIndexAdapter",
    "auto_adapt",
    "get_adapter",
]
