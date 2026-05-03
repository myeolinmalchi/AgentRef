"""End-to-end workflow scenarios for agentstate integration."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import agentstate
from agentstate import AgentState, Externalized, Framework, Inline
from agentstate.adapters.autogen import AutoGenAdapter
from agentstate.adapters.langgraph import LangGraphAdapter
from agentstate.adapters.llamaindex import LlamaIndexAdapter
from agentstate.core.invariants import validate_checkpoint_state
from agentstate.core.reference import ContentRef
from agentstate.storage import FilesystemCAS, InMemoryCAS


class E2EResearchState(AgentState):
    """State used across E2E workflow scenarios."""

    current_step: Inline[str]
    iteration: Inline[int]
    docs: Externalized[List[Dict[str, str]]]
    raw_html: Externalized[str]


def test_public_api_exports_only_documented_symbols() -> None:
    expected = {
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
    }

    assert set(agentstate.__all__) == expected
    for name in expected:
        assert hasattr(agentstate, name)


def test_langgraph_like_rag_workflow_keeps_payloads_out_of_checkpoints() -> None:
    backend = InMemoryCAS()
    agentstate.configure(backend=backend)
    adapter = LangGraphAdapter()
    raw_docs = [
        {"id": "doc-1", "text": "large retrieved document payload " * 16},
        {"id": "doc-2", "text": "another retrieved document payload " * 16},
    ]
    raw_html = "<html>" + ("expensive page body " * 32) + "</html>"

    initial = E2EResearchState(
        current_step="start",
        iteration=0,
        docs=[],
        raw_html="",
    ).to_checkpoint_dict()
    retrieve_update = adapter.externalize_node_update(
        E2EResearchState,
        {
            "current_step": "retrieve",
            "docs": raw_docs,
            "raw_html": raw_html,
        },
    )
    checkpoint_after_retrieve = dict(initial)
    checkpoint_after_retrieve.update(retrieve_update)
    validate_checkpoint_state(
        E2EResearchState,
        checkpoint_after_retrieve,
        require_externalized_exists=True,
    )

    retrieve_bytes = adapter.serialize_for_checkpoint(
        E2EResearchState.from_checkpoint_dict(checkpoint_after_retrieve)
    )

    assert raw_docs[0]["text"].encode() not in retrieve_bytes
    assert raw_html.encode() not in retrieve_bytes

    hydrated = adapter.hydrate_state_for_node(
        E2EResearchState,
        checkpoint_after_retrieve,
    )
    analyze_update = adapter.externalize_node_update(
        E2EResearchState,
        {
            "current_step": "analyze",
            "iteration": hydrated["iteration"] + len(hydrated["docs"]),
            "docs": raw_docs,
        },
    )
    checkpoint_after_analyze = dict(checkpoint_after_retrieve)
    checkpoint_after_analyze.update(analyze_update)

    old_state = adapter.deserialize_from_checkpoint(
        retrieve_bytes,
        E2EResearchState,
    )
    new_state = E2EResearchState.from_checkpoint_dict(checkpoint_after_analyze)

    assert old_state.current_step == "retrieve"
    assert old_state.docs == raw_docs
    assert old_state.raw_html == raw_html
    assert new_state.current_step == "analyze"
    assert new_state.iteration == 2
    assert checkpoint_after_retrieve["docs"] == checkpoint_after_analyze["docs"]


def test_llamaindex_like_context_workflow_round_trips_store_state() -> None:
    backend = InMemoryCAS()
    agentstate.configure(backend=backend)
    adapter = LlamaIndexAdapter()
    context_store: Dict[str, Any] = {}
    store = adapter.context_store(E2EResearchState, context_store)
    docs = [{"id": "paper", "text": "research workflow document " * 12}]

    store["current_step"] = "retrieve"
    store["iteration"] = 1
    store["docs"] = docs
    store["raw_html"] = "<main>workflow html</main>"

    checkpoint = store.to_checkpoint_dict()
    checkpoint_bytes = adapter.serialize_for_checkpoint(checkpoint)
    restored = adapter.deserialize_from_checkpoint(checkpoint_bytes, E2EResearchState)

    assert isinstance(checkpoint["docs"], ContentRef)
    assert docs[0]["text"].encode() not in checkpoint_bytes
    assert store["docs"] == docs
    assert restored.docs == docs
    assert restored.current_step == "retrieve"


def test_autogen_like_multi_agent_history_externalizes_tool_results() -> None:
    backend = InMemoryCAS()
    agentstate.configure(backend=backend)
    adapter = AutoGenAdapter()
    tool_result = "autogen multi-agent tool result " * 24
    messages = [
        {"role": "planner", "content": "fetch data"},
        {"role": "worker", "tool_result": tool_result},
    ]

    checkpoint_messages = adapter.externalize_message_history(
        messages,
        threshold_bytes=16,
    )
    checkpoint_bytes = pickle.dumps(checkpoint_messages)
    hydrated_messages = adapter.hydrate_message_history(checkpoint_messages)

    assert tool_result.encode() not in checkpoint_bytes
    assert isinstance(checkpoint_messages[1]["tool_result"], dict)
    assert hydrated_messages == messages


def test_filesystem_e2e_preserves_time_travel_across_backend_instances(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state_blobs"
    first_backend = FilesystemCAS(root=root, backend_id="shared-fs")
    agentstate.configure(backend=first_backend)
    adapter = LangGraphAdapter()
    docs = [{"id": "doc", "text": "filesystem-backed document payload"}]
    state = E2EResearchState(
        current_step="retrieve",
        iteration=1,
        docs=docs,
        raw_html="<html>snapshot</html>",
    )
    checkpoint_bytes = adapter.serialize_for_checkpoint(state)

    second_backend = FilesystemCAS(root=root, backend_id="shared-fs")
    agentstate.configure(backend=second_backend)
    restored = adapter.deserialize_from_checkpoint(
        checkpoint_bytes,
        E2EResearchState,
    )

    assert docs[0]["text"].encode() not in checkpoint_bytes
    assert restored.docs == docs
    assert restored.raw_html == "<html>snapshot</html>"
