"""Public API for agentref."""

from agentref.adapters import auto_adapt
from agentref.config import AgentRefRuntime, configure, create_runtime, get_config
from agentref.core.reference import ContentRef
from agentref.core.state import AgentRefState
from agentref.core.types import Externalized, Inline
from agentref.detection.framework import Framework, detect_active_framework
from agentref.exceptions import (
    AgentRefError,
    AmbiguousFrameworkError,
    InlineSizeExceeded,
    NoFrameworkDetectedError,
    UnresolvedReferenceError,
)

__all__ = [
    "AgentRefState",
    "AgentRefRuntime",
    "AgentRefError",
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
