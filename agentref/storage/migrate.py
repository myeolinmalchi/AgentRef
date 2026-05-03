"""Helpers for copying payloads between CAS backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from agentref.exceptions import AgentRefError
from agentref.storage.base import BaseCASBackend


@dataclass(frozen=True)
class MigrationResult:
    """Summary for a CAS migration run."""

    object_count: int
    bytes_copied: int


def migrate_cas(
    source: BaseCASBackend,
    target: BaseCASBackend,
    *,
    hashes: Optional[Iterable[str]] = None,
) -> MigrationResult:
    """Copy CAS objects from ``source`` to ``target``.

    When ``hashes`` is omitted, ``source`` must support ``iter_hashes()``.
    Existing checkpoint refs can continue to hydrate after migration when the
    target backend is configured with the source backend id as an alias.
    """

    selected_hashes = hashes if hashes is not None else source.iter_hashes()
    object_count = 0
    bytes_copied = 0
    for content_hash in selected_hashes:
        payload = source.get(content_hash)
        migrated_hash = target.put(payload)
        if migrated_hash != content_hash:
            raise AgentRefError(
                "CAS migration changed a content hash from "
                f"{content_hash!r} to {migrated_hash!r}."
            )
        object_count += 1
        bytes_copied += len(payload)
    return MigrationResult(object_count=object_count, bytes_copied=bytes_copied)
