"""Tests for the filesystem content-addressed storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentstate.core.reference import ContentRef
from agentstate.storage import FilesystemCAS


def test_filesystem_cas_stores_objects_in_sharded_paths(tmp_path: Path) -> None:
    backend = FilesystemCAS(root=tmp_path / "cas")
    payload = b"filesystem payload"

    content_hash = backend.put(payload)
    object_path = backend.path_for_hash(content_hash)

    assert object_path == backend.root / content_hash[:2] / content_hash[2:]
    assert object_path.read_bytes() == payload
    assert backend.get(content_hash) == payload
    assert backend.exists(content_hash)


def test_filesystem_cas_deduplicates_same_payload(tmp_path: Path) -> None:
    backend = FilesystemCAS(root=tmp_path / "cas")

    first = backend.put(b"same")
    path = backend.path_for_hash(first)
    first_stat = path.stat()
    second = backend.put(b"same")

    assert first == second
    assert path.stat().st_ino == first_stat.st_ino


def test_filesystem_cas_delete_is_idempotent(tmp_path: Path) -> None:
    backend = FilesystemCAS(root=tmp_path / "cas")
    content_hash = backend.put(b"payload")

    backend.delete(content_hash)
    backend.delete(content_hash)

    assert not backend.exists(content_hash)
    with pytest.raises(KeyError):
        backend.get(content_hash)


def test_filesystem_cas_round_trips_serialized_objects(tmp_path: Path) -> None:
    backend = FilesystemCAS(root=tmp_path / "cas")
    value = {"docs": [{"id": 1, "text": "hello"}]}
    payload = backend.serialize(value)
    content_hash = backend.put(payload)

    assert backend.deserialize(backend.get(content_hash), "dict") == value


def test_filesystem_cas_supports_content_ref_resolution(tmp_path: Path) -> None:
    backend = FilesystemCAS(root=tmp_path / "cas")
    payload = backend.serialize(["doc-a", "doc-b"])
    content_hash = backend.put(payload)
    ref = ContentRef(
        hash=content_hash,
        backend_id=backend.backend_id,
        type_name="list",
        size_bytes=len(payload),
    )

    assert ref.resolve(backend) == ["doc-a", "doc-b"]
