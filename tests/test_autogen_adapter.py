"""Tests for the AutoGen adapter."""

from __future__ import annotations

from pathlib import Path

from agentref.adapters.autogen import AutoGenAdapter
from agentref.config import configure
from agentref.core.reference import ContentRef
from agentref.core.state import AgentRefState
from agentref.core.types import Externalized, Inline
from agentref.storage import InMemoryCAS


class AutoGenResearchState(AgentRefState):
    """State class used by AutoGen adapter tests."""

    step: Inline[str]
    tool_result: Externalized[str]


def test_autogen_wrap_state_class_returns_original_class() -> None:
    adapter = AutoGenAdapter()

    assert adapter.wrap_state_class(AutoGenResearchState) is AutoGenResearchState


def test_autogen_bound_adapter_externalizes_and_hydrates_state() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = AutoGenAdapter(AutoGenResearchState)

    checkpoint = adapter.externalize_state(
        {"step": "tool", "tool_result": "large result"}
    )

    assert isinstance(checkpoint["tool_result"], ContentRef)
    assert adapter.hydrate_state(checkpoint) == {
        "step": "tool",
        "tool_result": "large result",
    }


def test_autogen_externalizes_and_hydrates_declared_state_mapping() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = AutoGenAdapter()

    checkpoint = adapter.externalize_state(
        AutoGenResearchState,
        {"step": "tool", "tool_result": "large result"},
    )

    assert checkpoint["step"] == "tool"
    assert isinstance(checkpoint["tool_result"], ContentRef)
    assert adapter.hydrate_state(AutoGenResearchState, checkpoint) == {
        "step": "tool",
        "tool_result": "large result",
    }


def test_autogen_message_history_externalizes_large_message_fields() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = AutoGenAdapter()
    raw = "autogen-tool-result-that-must-not-repeat"

    messages = adapter.externalize_message_history(
        [{"role": "assistant", "content": raw}],
        threshold_bytes=1,
    )

    content = messages[0]["content"]
    assert isinstance(content, dict)
    assert "agentref_ref" in content
    assert raw.encode() not in str(messages).encode()
    assert adapter.hydrate_message_history(messages)[0]["content"] == raw


def test_autogen_message_history_leaves_small_values_inline() -> None:
    adapter = AutoGenAdapter()

    messages = adapter.externalize_message_history(
        [{"role": "assistant", "content": "ok"}],
        threshold_bytes=100,
    )

    assert messages == [{"role": "assistant", "content": "ok"}]


def test_autogen_checkpoint_bytes_exclude_externalized_payload() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = AutoGenAdapter()
    raw_result = "autogen raw payload that must not be checkpointed"
    state = AutoGenResearchState(step="tool", tool_result=raw_result)

    checkpoint_bytes = adapter.serialize_for_checkpoint(state)

    assert raw_result.encode() not in checkpoint_bytes
    assert raw_result.encode() in backend.get(
        state.to_checkpoint_dict()["tool_result"].hash
    )


def test_autogen_deserializes_checkpoint_to_agent_ref() -> None:
    adapter = AutoGenAdapter()
    state = AutoGenResearchState(step="tool", tool_result="result")

    restored = adapter.deserialize_from_checkpoint(
        adapter.serialize_for_checkpoint(state),
        AutoGenResearchState,
    )

    assert restored.step == "tool"
    assert restored.tool_result == "result"


def test_autogen_limitation_document_exists() -> None:
    assert Path("docs/autogen_limitations.md").read_text().startswith(
        "# AutoGen Adapter Limitations"
    )
