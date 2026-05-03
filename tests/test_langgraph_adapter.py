"""Tests for the LangGraph adapter."""

from __future__ import annotations

import pickle
from typing import Any, Dict, get_args, get_origin

from agentref.adapters import auto_adapt
from agentref.adapters.langgraph import LangGraphAdapter
from agentref.config import configure
from agentref.core.reference import ContentRef
from agentref.core.reducers import ref_aware_replace
from agentref.core.state import AgentRefState
from agentref.core.types import Externalized, Inline
from agentref.detection.framework import Framework
from agentref.storage import InMemoryCAS


class LangGraphResearchState(AgentRefState):
    """State class used by LangGraph adapter tests."""

    step: Inline[str]
    docs: Externalized[list[str]]
    blob: Externalized[bytes]


def test_langgraph_adapter_wraps_state_class_without_langgraph_installed() -> None:
    adapter = LangGraphAdapter()

    schema = adapter.wrap_state_class(LangGraphResearchState)

    assert getattr(schema, "__agentref_origin__") is LangGraphResearchState
    assert schema.__annotations__["step"] is str
    docs_annotation = schema.__annotations__["docs"]
    assert get_origin(docs_annotation) is not None
    assert get_args(docs_annotation)[0] == Dict[str, Any]


def test_langgraph_bound_adapter_exposes_schema_and_wraps_nodes() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = LangGraphAdapter(LangGraphResearchState)

    schema = adapter.schema()

    assert getattr(schema, "__agentref_origin__") is LangGraphResearchState

    def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
        assert state["step"] == "retrieve"
        return {"docs": ["doc-a"], "blob": b"payload"}

    checkpoint_update = adapter.wrap_node(retrieve)({"step": "retrieve"})

    assert "agentref_ref" in checkpoint_update["docs"]
    assert "agentref_ref" in checkpoint_update["blob"]
    assert adapter.hydrate_state_for_node(checkpoint_update) == {
        "docs": ["doc-a"],
        "blob": b"payload",
    }


def test_langgraph_bound_adapter_can_take_backend_without_global_configure() -> None:
    backend = InMemoryCAS("adapter-local")
    adapter = LangGraphAdapter(LangGraphResearchState, backend=backend)

    update = adapter.externalize_node_update({"docs": ["doc-a"], "blob": b"payload"})

    assert backend.object_count == 2
    assert update["docs"]["agentref_ref"]["backend_id"] == "adapter-local"
    assert adapter.hydrate_state_for_node(update) == {
        "docs": ["doc-a"],
        "blob": b"payload",
    }


def test_langgraph_adapter_installs_reducers_for_externalized_fields() -> None:
    adapter = LangGraphAdapter()

    reducers = adapter.install_reducers(LangGraphResearchState)

    assert set(reducers) == {"docs", "blob"}
    assert reducers["docs"] is ref_aware_replace
    assert reducers["blob"] is ref_aware_replace


def test_langgraph_externalizes_node_update_and_hydrates_state() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = LangGraphAdapter()

    update = adapter.externalize_node_update(
        LangGraphResearchState,
        {"step": "retrieve", "docs": ["doc-a"], "blob": b"large"},
    )

    assert update["step"] == "retrieve"
    assert "agentref_ref" in update["docs"]
    assert "agentref_ref" in update["blob"]
    assert backend.object_count == 2
    assert adapter.hydrate_state_for_node(LangGraphResearchState, update) == {
        "step": "retrieve",
        "docs": ["doc-a"],
        "blob": b"large",
    }


def test_langgraph_checkpoint_bytes_exclude_externalized_payload() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = LangGraphAdapter()
    raw_blob = b"langgraph-raw-payload-that-must-not-repeat"
    state = LangGraphResearchState(step="retrieve", docs=["doc"], blob=raw_blob)

    checkpoint_bytes = adapter.serialize_for_checkpoint(state)

    assert raw_blob not in checkpoint_bytes
    assert raw_blob in backend.get(state.to_checkpoint_dict()["blob"].hash)


def test_langgraph_deserializes_checkpoint_to_agent_ref() -> None:
    adapter = LangGraphAdapter()
    state = LangGraphResearchState(step="retrieve", docs=["doc"], blob=b"payload")

    restored = adapter.deserialize_from_checkpoint(
        adapter.serialize_for_checkpoint(state),
        LangGraphResearchState,
    )

    assert restored.step == "retrieve"
    assert restored.docs == ["doc"]
    assert restored.blob == b"payload"


def test_auto_adapt_returns_langgraph_schema_for_explicit_framework() -> None:
    schema = auto_adapt(LangGraphResearchState, Framework.LANGGRAPH)

    assert getattr(schema, "__agentref_origin__") is LangGraphResearchState


def test_langgraph_serialize_accepts_checkpoint_mapping() -> None:
    adapter = LangGraphAdapter()
    state = LangGraphResearchState(step="retrieve", docs=["doc"], blob=b"payload")
    checkpoint = state.to_checkpoint_dict()

    loaded = pickle.loads(adapter.serialize_for_checkpoint(checkpoint))

    assert loaded == checkpoint
    assert isinstance(loaded["docs"], ContentRef)


def test_langgraph_mapping_helpers_reject_unknown_fields() -> None:
    adapter = LangGraphAdapter()

    try:
        adapter.externalize_node_update(
            LangGraphResearchState,
            {"missing": "value"},
        )
    except KeyError as exc:
        assert "Unknown field" in str(exc)
    else:
        raise AssertionError("unknown fields should be rejected")
