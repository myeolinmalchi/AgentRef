"""Tests for AgentStateMeta and AgentState behavior."""

from __future__ import annotations

import pytest

from agentstate.config import configure
from agentstate.core.descriptors import ExternalizedDescriptor, InlineDescriptor
from agentstate.core.reference import ContentRef
from agentstate.core.state import AgentState
from agentstate.core.types import Externalized, Inline
from agentstate.exceptions import AgentStateError
from agentstate.storage import InMemoryCAS


def test_metaclass_installs_descriptors_and_field_metadata() -> None:
    class ResearchState(AgentState):
        step: Inline[str]
        docs: Externalized[list[str]]

    assert isinstance(ResearchState.__dict__["step"], InlineDescriptor)
    assert isinstance(ResearchState.__dict__["docs"], ExternalizedDescriptor)
    assert ResearchState.fields()["step"].kind == "inline"
    assert ResearchState.fields()["docs"].kind == "externalized"
    assert ResearchState.inline_fields().keys() == {"step"}
    assert ResearchState.externalized_fields().keys() == {"docs"}


def test_metaclass_preserves_inherited_agentstate_fields() -> None:
    class BaseState(AgentState):
        current_step: Inline[str]

    class ChildState(BaseState):
        docs: Externalized[list[str]]

    assert set(ChildState.fields()) == {"current_step", "docs"}
    child = ChildState(current_step="retrieve", docs=["a"])

    assert child.current_step == "retrieve"
    assert child.docs == ["a"]


def test_constructor_rejects_unknown_fields() -> None:
    class ResearchState(AgentState):
        step: Inline[str]

    with pytest.raises(AgentStateError, match="Unknown field"):
        ResearchState(step="retrieve", missing=True)


def test_unassigned_field_access_raises_attribute_error() -> None:
    class ResearchState(AgentState):
        step: Inline[str]
        docs: Externalized[list[str]]

    state = ResearchState()

    with pytest.raises(AttributeError, match="step"):
        _ = state.step
    with pytest.raises(AttributeError, match="docs"):
        _ = state.docs


def test_checkpoint_round_trip_restores_inline_and_externalized_fields() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)

    class ResearchState(AgentState):
        step: Inline[str]
        docs: Externalized[list[str]]

    original = ResearchState(step="retrieve", docs=["doc-a", "doc-b"])
    checkpoint = original.to_checkpoint_dict()
    restored = ResearchState.from_checkpoint_dict(checkpoint)

    assert restored.step == "retrieve"
    assert restored.docs == ["doc-a", "doc-b"]
    assert restored.to_checkpoint_dict() == checkpoint


def test_checkpoint_round_trip_accepts_content_ref_dicts() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)

    class BlobState(AgentState):
        blob: Externalized[bytes]

    original = BlobState(blob=b"payload")
    ref = original.to_checkpoint_dict()["blob"]
    assert isinstance(ref, ContentRef)

    restored = BlobState.from_checkpoint_dict({"blob": ref.to_dict()})

    assert restored.blob == b"payload"


def test_checkpoint_restore_rejects_invalid_externalized_value() -> None:
    class BlobState(AgentState):
        blob: Externalized[bytes]

    with pytest.raises(AgentStateError, match="ContentRef"):
        BlobState.from_checkpoint_dict({"blob": b"not-a-ref"})


def test_mapping_access_hydrates_and_assigns_declared_fields() -> None:
    class ResearchState(AgentState):
        step: Inline[str]
        docs: Externalized[list[str]]

    state = ResearchState()
    state["step"] = "retrieve"
    state["docs"] = ["doc"]

    assert state["step"] == "retrieve"
    assert state["docs"] == ["doc"]
    assert "step" in state
    assert "missing" not in state

    with pytest.raises(KeyError, match="Unknown field"):
        _ = state["missing"]


def test_framework_conversion_methods_are_checkpoint_safe_aliases() -> None:
    class ResearchState(AgentState):
        step: Inline[str]
        docs: Externalized[list[str]]

    state = ResearchState(step="retrieve", docs=["doc"])
    checkpoint = state.to_checkpoint_dict()

    assert state.to_langgraph_state() == checkpoint
    assert state.to_llamaindex_context_dict() == checkpoint
    assert state.to_autogen_state() == checkpoint
    assert ResearchState.from_langgraph_state(checkpoint).docs == ["doc"]
    assert ResearchState.from_llamaindex_context_dict(checkpoint).docs == ["doc"]
    assert ResearchState.from_autogen_state(checkpoint).docs == ["doc"]
