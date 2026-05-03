"""In-memory content-addressed storage backend."""

from __future__ import annotations

from typing import Dict

from agentstate.storage.base import BaseCASBackend


class InMemoryCAS(BaseCASBackend):
    """Dictionary-backed CAS backend for tests and ephemeral workflows."""

    def __init__(self, backend_id: str = "memory") -> None:
        """Initialize an empty in-memory backend."""

        self._backend_id = backend_id
        self._store: Dict[str, bytes] = {}

    @property
    def backend_id(self) -> str:
        """Stable identifier for this backend instance."""

        return self._backend_id

    def put(self, data: bytes) -> str:
        """Store bytes by SHA-256 hash and return that hash."""

        content_hash = self.hash_bytes(data)
        self._store.setdefault(content_hash, bytes(data))
        return content_hash

    def get(self, hash: str) -> bytes:
        """Return bytes for ``hash``.

        Raises:
            KeyError: If ``hash`` does not exist.
        """

        return self._store[hash]

    def exists(self, hash: str) -> bool:
        """Return whether ``hash`` exists in storage."""

        return hash in self._store

    def delete(self, hash: str) -> None:
        """Delete ``hash`` from storage if present."""

        self._store.pop(hash, None)

    @property
    def object_count(self) -> int:
        """Return the number of unique objects stored."""

        return len(self._store)
