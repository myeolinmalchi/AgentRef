"""Tests for the LlamaIndex adapter."""

from __future__ import annotations

import pickle
from typing import Any, Dict

from agentstate.adapters.llamaindex import LlamaIndexAdapter, LlamaIndexStateSpec
from agentstate.config import configure
from agentstate.core.reference import ContentRef
from agentstate.core.state import AgentState
from agentstate.core.types import Externalized, Inline
from agentstate.storage import InMemoryCAS


class LlamaIndexResearchState(AgentState):
    """State class used by LlamaIndex adapter tests."""

    step: Inline[str]
    docs: Externalized[list[str]]


def test_llamaindex_wrap_state_class_returns_context_store_spec() -> None:
    adapter = LlamaIndexAdapter()

    spec = adapter.wrap_state_class(LlamaIndexResearchState)

    assert isinstance(spec, LlamaIndexStateSpec)
    assert spec.state_cls is LlamaIndexResearchState
    assert spec.fields == {"step": "inline", "docs": "externalized"}


def test_llamaindex_bound_adapter_wraps_context_store() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = LlamaIndexAdapter(LlamaIndexResearchState)
    raw_store: Dict[str, Any] = {}

    store = adapter.context_store(raw_store)
    store["docs"] = ["doc-a"]

    assert isinstance(raw_store["docs"], ContentRef)
    assert store["docs"] == ["doc-a"]


def test_llamaindex_context_store_proxy_externalizes_and_hydrates() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = LlamaIndexAdapter()
    raw_store: Dict[str, Any] = {}
    store = adapter.context_store(LlamaIndexResearchState, raw_store)

    store["step"] = "retrieve"
    store["docs"] = ["doc-a", "doc-b"]

    assert raw_store["step"] == "retrieve"
    assert isinstance(raw_store["docs"], ContentRef)
    assert store["docs"] == ["doc-a", "doc-b"]
    assert store.get("docs") == ["doc-a", "doc-b"]
    assert store.to_checkpoint_dict() == raw_store


def test_llamaindex_checkpoint_bytes_exclude_externalized_payload() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = LlamaIndexAdapter()
    raw_docs = ["llamaindex-raw-doc-that-must-not-repeat"]
    state = LlamaIndexResearchState(step="retrieve", docs=raw_docs)

    checkpoint_bytes = adapter.serialize_for_checkpoint(state)

    assert raw_docs[0].encode() not in checkpoint_bytes
    assert raw_docs[0].encode() in backend.get(state.to_checkpoint_dict()["docs"].hash)


def test_llamaindex_deserializes_checkpoint_to_agent_state() -> None:
    adapter = LlamaIndexAdapter()
    state = LlamaIndexResearchState(step="retrieve", docs=["doc-a"])

    restored = adapter.deserialize_from_checkpoint(
        adapter.serialize_for_checkpoint(state),
        LlamaIndexResearchState,
    )

    assert restored.step == "retrieve"
    assert restored.docs == ["doc-a"]


def test_llamaindex_serialize_accepts_context_mapping() -> None:
    adapter = LlamaIndexAdapter()
    state = LlamaIndexResearchState(step="retrieve", docs=["doc-a"])

    loaded = pickle.loads(adapter.serialize_for_checkpoint(state.to_checkpoint_dict()))

    assert loaded == state.to_checkpoint_dict()
