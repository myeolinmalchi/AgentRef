"""Tests for Inline/Externalized marker types and descriptors."""

from __future__ import annotations

import pickle
from typing import get_args, get_origin

import pytest

from agentref.config import configure, get_config
from agentref.core.reference import ContentRef
from agentref.core.state import AgentRefState
from agentref.core.types import (
    Externalized,
    Inline,
    get_wrapped_type,
    is_externalized_annotation,
    is_inline_annotation,
)
from agentref.exceptions import AgentRefError, InlineSizeExceeded
from agentref.storage import InMemoryCAS


def test_inline_and_externalized_are_runtime_generic_markers() -> None:
    inline_annotation = Inline[int]
    externalized_annotation = Externalized[list[dict[str, int]]]

    assert get_origin(inline_annotation) is Inline
    assert get_args(inline_annotation) == (int,)
    assert is_inline_annotation(inline_annotation)
    assert not is_externalized_annotation(inline_annotation)
    assert is_externalized_annotation(externalized_annotation)
    assert get_wrapped_type(externalized_annotation) == list[dict[str, int]]


def test_configure_updates_backend_threshold_and_framework() -> None:
    backend = InMemoryCAS("custom")

    config = configure(
        backend=backend,
        inline_threshold_bytes=128,
        framework="langgraph",
    )

    assert config is get_config()
    assert get_config().backend is backend
    assert get_config().inline_threshold_bytes == 128
    assert get_config().framework == "langgraph"


def test_configure_rejects_non_positive_inline_threshold() -> None:
    with pytest.raises(AgentRefError, match="positive integer"):
        configure(inline_threshold_bytes=0)


def test_inline_descriptor_uses_runtime_threshold_after_class_definition() -> None:
    class TinyState(AgentRefState):
        payload: Inline[bytes]

    state = TinyState()
    configure(inline_threshold_bytes=3)

    state.payload = b"abc"
    assert state.payload == b"abc"

    with pytest.raises(InlineSizeExceeded, match="payload"):
        state.payload = b"abcd"


def test_externalized_descriptor_stores_content_ref_and_hydrates_value() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)

    class ResearchState(AgentRefState):
        docs: Externalized[list[dict[str, int]]]

    value = [{"id": 1}, {"id": 2}]
    state = ResearchState(docs=value)
    checkpoint = state.to_checkpoint_dict()

    assert isinstance(checkpoint["docs"], ContentRef)
    assert checkpoint["docs"].backend_id == backend.backend_id
    assert backend.object_count == 1
    assert state.docs == value


def test_externalized_descriptor_deduplicates_same_data() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)

    class ResearchState(AgentRefState):
        docs: Externalized[list[str]]

    first = ResearchState(docs=["same"])
    second = ResearchState(docs=["same"])

    assert first.to_checkpoint_dict()["docs"] == second.to_checkpoint_dict()["docs"]
    assert backend.object_count == 1


def test_externalized_payload_never_appears_in_checkpoint_bytes_after_hydration() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)

    class BlobState(AgentRefState):
        blob: Externalized[bytes]

    raw_blob = b"unique-binary-payload-that-must-not-be-checkpointed"
    state = BlobState(blob=raw_blob)

    assert state.blob == raw_blob
    checkpoint_bytes = pickle.dumps(state.to_checkpoint_dict())

    assert raw_blob not in checkpoint_bytes
    ref = state.to_checkpoint_dict()["blob"]
    assert isinstance(ref, ContentRef)
    assert backend.get(ref.hash) == raw_blob


def test_externalized_descriptor_accepts_existing_content_ref_for_restore() -> None:
    backend = InMemoryCAS()
    configure(backend=backend)

    class BlobState(AgentRefState):
        blob: Externalized[bytes]

    original = BlobState(blob=b"payload")
    ref = original.to_checkpoint_dict()["blob"]
    restored = BlobState(blob=ref)

    assert isinstance(ref, ContentRef)
    assert restored.to_checkpoint_dict()["blob"] is ref
    assert restored.blob == b"payload"
