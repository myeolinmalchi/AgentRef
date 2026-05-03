"""Real AutoGen message/state compatibility checks for AgentState adapters."""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Tuple

import pytest

from agentstate import AgentState, Externalized, Inline, configure
from agentstate.adapters.autogen import AutoGenAdapter
from agentstate.storage import InMemoryCAS


RAW_TOOL_RESULT = "autogen compatibility tool result " * 4096
RAW_AGENT_CONTENT = "autogen compatibility assistant content " * 4096


class AutoGenCompatState(AgentState):
    """AgentState schema for explicit AutoGen state-dict integration."""

    phase: Inline[str]
    tool_result: Externalized[str]
    transcript: Externalized[List[Dict[str, str]]]


def _autogen_imports() -> Tuple[Any, Any, Any]:
    """Import current AutoGen classes, skipping older incompatible packages."""

    messages_mod = pytest.importorskip("autogen_agentchat.messages")
    models_mod = pytest.importorskip("autogen_core.models")
    return (
        messages_mod.TextMessage,
        messages_mod.ToolCallExecutionEvent,
        models_mod.FunctionExecutionResult,
    )


def _message_dict(message: Any) -> Dict[str, Any]:
    """Return a JSON-like dict from a Pydantic AutoGen message."""

    if hasattr(message, "model_dump"):
        return dict(message.model_dump(mode="json"))
    return dict(message.dict())


@pytest.mark.compatibility
def test_autogen_message_history_externalizes_single_and_multi_agent_content() -> None:
    """Check single-agent and two-agent message histories without an LLM client."""

    TextMessage, _, _ = _autogen_imports()
    adapter = AutoGenAdapter()
    backend = InMemoryCAS()
    configure(backend=backend)

    single_agent_message = TextMessage(source="assistant", content=RAW_AGENT_CONTENT)
    multi_agent_messages = [
        TextMessage(source="planner", content="plan"),
        TextMessage(source="worker", content=RAW_AGENT_CONTENT),
    ]
    messages = [_message_dict(single_agent_message), *map(_message_dict, multi_agent_messages)]

    checkpoint_messages = adapter.externalize_message_history(
        messages,
        threshold_bytes=128,
    )
    hydrated_messages = adapter.hydrate_message_history(checkpoint_messages)
    checkpoint_bytes = pickle.dumps(checkpoint_messages)

    assert RAW_AGENT_CONTENT.encode() not in checkpoint_bytes
    assert hydrated_messages[0]["content"] == RAW_AGENT_CONTENT
    assert hydrated_messages[2]["content"] == RAW_AGENT_CONTENT
    assert [message["source"] for message in hydrated_messages] == [
        "assistant",
        "planner",
        "worker",
    ]
    assert backend.object_count == 1


@pytest.mark.compatibility
def test_autogen_tool_execution_event_payload_is_externalized() -> None:
    """Check large tool-call results using current AutoGen message models."""

    _, ToolCallExecutionEvent, FunctionExecutionResult = _autogen_imports()
    adapter = AutoGenAdapter()
    configure(backend=InMemoryCAS())

    event = ToolCallExecutionEvent(
        source="tool",
        content=[
            FunctionExecutionResult(
                content=RAW_TOOL_RESULT,
                name="large_lookup",
                call_id="call-1",
                is_error=False,
            )
        ],
    )
    messages = [_message_dict(event)]

    checkpoint_messages = adapter.externalize_message_history(
        messages,
        threshold_bytes=128,
    )
    checkpoint_bytes = pickle.dumps(checkpoint_messages)
    hydrated = adapter.hydrate_message_history(checkpoint_messages)

    assert RAW_TOOL_RESULT.encode() not in checkpoint_bytes
    assert hydrated[0]["content"][0]["content"] == RAW_TOOL_RESULT
    assert hydrated[0]["content"][0]["name"] == "large_lookup"


@pytest.mark.compatibility
def test_autogen_explicit_state_dict_round_trips_large_tool_result() -> None:
    """Check the Phase 7.5 partial integration scope for state dictionaries."""

    _autogen_imports()
    adapter = AutoGenAdapter()
    backend = InMemoryCAS()
    configure(backend=backend)
    transcript = [{"source": "planner", "content": "plan"}]

    checkpoint_state = adapter.externalize_state(
        AutoGenCompatState,
        {
            "phase": "tool",
            "tool_result": RAW_TOOL_RESULT,
            "transcript": transcript,
        },
    )
    checkpoint_bytes = adapter.serialize_for_checkpoint(checkpoint_state)
    hydrated = adapter.hydrate_state(AutoGenCompatState, checkpoint_state)

    assert RAW_TOOL_RESULT.encode() not in checkpoint_bytes
    assert hydrated == {
        "phase": "tool",
        "tool_result": RAW_TOOL_RESULT,
        "transcript": transcript,
    }
    assert backend.object_count == 2
