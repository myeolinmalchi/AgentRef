"""Public API for agentstate."""

from agentstate.adapters import auto_adapt
from agentstate.config import AgentStateRuntime, configure, create_runtime, get_config
from agentstate.core.reference import ContentRef
from agentstate.core.state import AgentState
from agentstate.core.types import Externalized, Inline
from agentstate.detection.framework import Framework, detect_active_framework
from agentstate.exceptions import (
    AgentStateError,
    AmbiguousFrameworkError,
    InlineSizeExceeded,
    NoFrameworkDetectedError,
    UnresolvedReferenceError,
)

__all__ = [
    "AgentState",
    "AgentStateRuntime",
    "AgentStateError",
    "AmbiguousFrameworkError",
    "ContentRef",
    "Externalized",
    "Framework",
    "Inline",
    "InlineSizeExceeded",
    "NoFrameworkDetectedError",
    "UnresolvedReferenceError",
    "auto_adapt",
    "configure",
    "create_runtime",
    "detect_active_framework",
    "get_config",
]
