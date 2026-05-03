"""Detection of the active agent framework."""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Set

from agentref.config import get_config
from agentref.exceptions import (
    AgentRefError,
    AmbiguousFrameworkError,
    NoFrameworkDetectedError,
)


class Framework(Enum):
    """Supported framework identifiers."""

    LANGGRAPH = "langgraph"
    LLAMAINDEX = "llamaindex"
    AUTOGEN = "autogen"


_FRAMEWORK_MODULES: Dict[Framework, tuple[str, ...]] = {
    Framework.LANGGRAPH: ("langgraph",),
    Framework.LLAMAINDEX: ("llama_index", "llama_index.core"),
    Framework.AUTOGEN: ("autogen", "autogen_agentchat"),
}


def detect_active_framework(explicit: Optional[Framework] = None) -> Framework:
    """Detect the active framework using explicit, configured, then import state.

    Args:
        explicit: Optional explicit framework selection. This wins over global
            configuration and import-based detection.

    Raises:
        AmbiguousFrameworkError: If more than one supported framework appears
            imported.
        NoFrameworkDetectedError: If no supported framework appears imported.
        AgentRefError: If an explicit or configured framework value is invalid.
    """

    if explicit is not None:
        return _coerce_framework(explicit)

    configured = get_config().framework
    if configured is not None:
        return _coerce_framework(configured)

    active = _active_frameworks_from_modules(sys.modules.keys())
    if len(active) == 1:
        return next(iter(active))
    if not active:
        supported = ", ".join(framework.value for framework in Framework)
        raise NoFrameworkDetectedError(
            "No active framework detected. Import one supported framework "
            f"({supported}) or call agentref.configure(framework=...)."
        )

    names = ", ".join(sorted(framework.value for framework in active))
    raise AmbiguousFrameworkError(
        "Multiple active frameworks detected: "
        f"{names}. Pass an explicit framework or call "
        "agentref.configure(framework=...)."
    )


def _coerce_framework(value: Any) -> Framework:
    """Convert enum or string values to ``Framework``."""

    if isinstance(value, Framework):
        return value
    if isinstance(value, str):
        normalized = value.replace("-", "_").lower()
        aliases = {
            "langgraph": Framework.LANGGRAPH,
            "llamaindex": Framework.LLAMAINDEX,
            "llama_index": Framework.LLAMAINDEX,
            "llama_index_core": Framework.LLAMAINDEX,
            "autogen": Framework.AUTOGEN,
            "pyautogen": Framework.AUTOGEN,
            "autogen_agentchat": Framework.AUTOGEN,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise AgentRefError(
                f"Unknown framework {value!r}. Supported frameworks: "
                f"{', '.join(framework.value for framework in Framework)}."
            ) from exc
    raise AgentRefError(
        f"Framework must be a Framework enum or string, found {type(value).__name__}."
    )


def _active_frameworks_from_modules(module_names: Iterable[str]) -> Set[Framework]:
    """Return frameworks whose root modules are present in ``module_names``."""

    names = set(module_names)
    active: Set[Framework] = set()
    for framework, module_roots in _FRAMEWORK_MODULES.items():
        if any(_module_loaded(names, root) for root in module_roots):
            active.add(framework)
    return active


def _module_loaded(module_names: Set[str], root: str) -> bool:
    """Return whether ``root`` or one of its submodules is loaded."""

    prefix = f"{root}."
    return any(name == root or name.startswith(prefix) for name in module_names)
