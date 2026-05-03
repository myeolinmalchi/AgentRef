"""Tests for Phase 1 content-addressed storage primitives."""

from __future__ import annotations

import pickle

import pytest

from agentref.core.reference import ContentRef
from agentref.exceptions import SerializationError, UnresolvedReferenceError
from agentref.storage import InMemoryCAS


def _store_object(backend: InMemoryCAS, value: object) -> ContentRef:
    payload = backend.serialize(value)
    content_hash = backend.put(payload)
    return ContentRef(
        hash=content_hash,
        backend_id=backend.backend_id,
        type_name=type(value).__name__,
        size_bytes=len(payload),
    )


def test_same_data_produces_same_hash_and_deduplicates() -> None:
    backend = InMemoryCAS()
    first = backend.put(b"same payload")
    second = backend.put(b"same payload")

    assert first == second
    assert backend.object_count == 1
    assert backend.get(first) == b"same payload"


def test_different_data_produces_different_hashes() -> None:
    backend = InMemoryCAS()

    assert backend.put(b"left") != backend.put(b"right")


def test_get_missing_hash_raises_key_error() -> None:
    backend = InMemoryCAS()

    with pytest.raises(KeyError):
        backend.get("missing")


def test_delete_removes_hash() -> None:
    backend = InMemoryCAS()
    content_hash = backend.put(b"payload")

    backend.delete(content_hash)

    assert not backend.exists(content_hash)


def test_serialize_bytes_passes_through() -> None:
    backend = InMemoryCAS()

    assert backend.serialize(b"raw bytes") == b"raw bytes"
    assert backend.deserialize(b"raw bytes", "bytes") == b"raw bytes"


def test_serialize_round_trips_common_python_objects() -> None:
    backend = InMemoryCAS()
    value = {"docs": [{"id": 1, "text": "hello"}], "scores": [1, 2, 3]}
    payload = backend.serialize(value)

    assert backend.deserialize(payload, "dict") == value


def test_content_ref_equality_and_hash_use_content_hash_only() -> None:
    left = ContentRef(
        hash="abc",
        backend_id="memory-a",
        type_name="str",
        size_bytes=3,
    )
    right = ContentRef(
        hash="abc",
        backend_id="memory-b",
        type_name="bytes",
        size_bytes=10,
    )

    assert left == right
    assert len({left, right}) == 1


def test_content_ref_serializes_to_json_dict_and_pickle_without_payload() -> None:
    backend = InMemoryCAS()
    ref = _store_object(backend, {"secret": "value"})
    assert ref.resolve(backend) == {"secret": "value"}
    assert ref.is_resolved

    restored_from_dict = ContentRef.from_dict(ref.to_dict())
    restored_from_json = ContentRef.from_json(ref.to_json())
    restored_from_pickle = pickle.loads(pickle.dumps(ref))

    assert restored_from_dict == ref
    assert restored_from_json == ref
    assert restored_from_pickle == ref
    assert not restored_from_pickle.is_resolved


def test_content_ref_serializes_to_msgpack_when_available() -> None:
    pytest.importorskip("msgpack")
    ref = ContentRef(
        hash="abc",
        backend_id="memory",
        type_name="str",
        size_bytes=3,
    )

    restored = ContentRef.from_msgpack(ref.to_msgpack())

    assert restored == ref
    assert restored.to_dict() == ref.to_dict()


def test_content_ref_resolves_lazily_from_backend() -> None:
    backend = InMemoryCAS()
    ref = _store_object(backend, ["doc-1", "doc-2"])

    assert not ref.is_resolved
    assert ref.resolve(backend) == ["doc-1", "doc-2"]
    assert ref.is_resolved


def test_content_ref_resolve_rejects_wrong_backend() -> None:
    backend = InMemoryCAS("memory-a")
    wrong_backend = InMemoryCAS("memory-b")
    ref = _store_object(backend, "payload")

    with pytest.raises(UnresolvedReferenceError, match="does not match"):
        ref.resolve(wrong_backend)


def test_content_ref_resolve_missing_content_raises_clear_error() -> None:
    backend = InMemoryCAS()
    ref = _store_object(backend, "payload")
    backend.delete(ref.hash)

    with pytest.raises(UnresolvedReferenceError, match="content is missing"):
        ref.resolve(backend)


def test_content_ref_from_dict_requires_all_fields() -> None:
    with pytest.raises(SerializationError, match="missing required key"):
        ContentRef.from_dict({"hash": "abc"})
