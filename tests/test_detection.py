"""Tests for active framework detection."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Iterable, cast

import pytest

from agentstate.config import configure
from agentstate.core.state import AgentState
from agentstate.core.types import Externalized, Inline
from agentstate.detection.framework import Framework, detect_active_framework
from agentstate.exceptions import (
    AgentStateError,
    AmbiguousFrameworkError,
    NoFrameworkDetectedError,
)


_FRAMEWORK_ROOTS = ("langgraph", "llama_index", "autogen", "autogen_agentchat")


def _clear_framework_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove supported framework modules from ``sys.modules`` for one test."""

    for name in list(sys.modules):
        if _matches_any_root(name, _FRAMEWORK_ROOTS):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _matches_any_root(name: str, roots: Iterable[str]) -> bool:
    """Return whether ``name`` is a root module or submodule of any root."""

    return any(name == root or name.startswith(f"{root}.") for root in roots)


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Install a placeholder imported module for detection tests."""

    monkeypatch.setitem(sys.modules, name, ModuleType(name))


def test_detect_active_framework_uses_explicit_argument_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_framework_modules(monkeypatch)
    _install_module(monkeypatch, "langgraph")

    assert detect_active_framework(Framework.AUTOGEN) is Framework.AUTOGEN


def test_detect_active_framework_uses_configured_framework_before_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_framework_modules(monkeypatch)
    _install_module(monkeypatch, "langgraph")
    configure(framework="llama_index")

    assert detect_active_framework() is Framework.LLAMAINDEX


def test_detect_active_framework_detects_single_imported_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_framework_modules(monkeypatch)
    _install_module(monkeypatch, "langgraph")

    assert detect_active_framework() is Framework.LANGGRAPH


def test_detect_active_framework_detects_submodules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_framework_modules(monkeypatch)
    _install_module(monkeypatch, "llama_index.core.workflow")

    assert detect_active_framework() is Framework.LLAMAINDEX


def test_detect_active_framework_detects_autogen_agentchat_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_framework_modules(monkeypatch)
    _install_module(monkeypatch, "autogen_agentchat")

    assert detect_active_framework() is Framework.AUTOGEN


def test_detect_active_framework_raises_when_none_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_framework_modules(monkeypatch)

    with pytest.raises(NoFrameworkDetectedError, match="No active framework"):
        detect_active_framework()


def test_detect_active_framework_raises_when_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_framework_modules(monkeypatch)
    _install_module(monkeypatch, "langgraph")
    _install_module(monkeypatch, "llama_index")

    with pytest.raises(AmbiguousFrameworkError, match="Multiple active frameworks"):
        detect_active_framework()


def test_detect_active_framework_rejects_unknown_configured_value() -> None:
    configure(framework="unknown")

    with pytest.raises(AgentStateError, match="Unknown framework"):
        detect_active_framework()


def test_detect_active_framework_rejects_invalid_configured_type() -> None:
    configure(framework=object())

    with pytest.raises(AgentStateError, match="Framework must"):
        detect_active_framework()


def test_dispatch_to_returns_checkpoint_safe_mapping() -> None:
    class ResearchState(AgentState):
        step: Inline[str]
        docs: Externalized[list[str]]

    state = ResearchState(step="retrieve", docs=["doc-a"])

    dispatched = state.dispatch_to(Framework.LANGGRAPH)

    assert dispatched == state.to_checkpoint_dict()
    assert dispatched["step"] == "retrieve"
    assert dispatched["docs"] == state.to_checkpoint_dict()["docs"]


def test_dispatch_to_accepts_string_values_at_runtime() -> None:
    class ResearchState(AgentState):
        step: Inline[str]

    state = ResearchState(step="retrieve")

    assert state.dispatch_to(cast(Any, "autogen")) == state.to_autogen_state()
