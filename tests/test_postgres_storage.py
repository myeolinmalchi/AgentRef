"""Tests for PostgreSQL CAS behavior without a live database."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pytest

from agentstate.core.reference import ContentRef
from agentstate.storage import InMemoryCAS, PostgresCAS, migrate_cas


class FakeCursor:
    """Small cursor stand-in for PostgresCAS unit tests."""

    def __init__(
        self,
        rows: Optional[Iterable[Tuple[Any, ...]]] = None,
        rowcount: int = 0,
    ) -> None:
        self._rows = list(rows or [])
        self.rowcount = rowcount

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        """Return one row or ``None``."""

        if not self._rows:
            return None
        return self._rows.pop(0)

    def __iter__(self) -> Iterator[Tuple[Any, ...]]:
        """Iterate remaining rows."""

        return iter(self._rows)


class FakePostgresConnection:
    """Minimal psycopg3-style connection for CAS tests."""

    def __init__(self) -> None:
        self.records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.commit_count = 0

    def execute(self, sql: str, params: Sequence[Any] = ()) -> FakeCursor:
        """Execute the subset of SQL emitted by ``PostgresCAS``."""

        normalized = " ".join(sql.split()).upper()
        if normalized.startswith("INSERT INTO"):
            backend_id, content_hash, payload, size_bytes, expires_at = params
            key = (str(backend_id), str(content_hash))
            if key not in self.records:
                self.records[key] = {
                    "payload": bytes(payload),
                    "size_bytes": int(size_bytes),
                    "expires_at": expires_at,
                }
            elif expires_at is not None:
                self.records[key]["expires_at"] = expires_at
            return FakeCursor(rowcount=1)

        if normalized.startswith("SELECT PAYLOAD"):
            backend_id, content_hash = str(params[0]), str(params[1])
            record = self.records.get((backend_id, content_hash))
            if record is None or _is_expired(record):
                return FakeCursor()
            return FakeCursor([(record["payload"],)])

        if normalized.startswith("SELECT 1"):
            backend_id, content_hash = str(params[0]), str(params[1])
            record = self.records.get((backend_id, content_hash))
            return FakeCursor([(1,)] if record is not None and not _is_expired(record) else [])

        if normalized.startswith("SELECT HASH"):
            backend_id = str(params[0])
            rows = [
                (content_hash,)
                for (record_backend_id, content_hash), record in sorted(
                    self.records.items()
                )
                if record_backend_id == backend_id and not _is_expired(record)
            ]
            return FakeCursor(rows)

        if normalized.startswith("DELETE FROM") and "AND HASH" in normalized:
            backend_id, content_hash = str(params[0]), str(params[1])
            removed = self.records.pop((backend_id, content_hash), None)
            return FakeCursor(rowcount=1 if removed is not None else 0)

        if normalized.startswith("DELETE FROM") or normalized.startswith("WITH DOOMED"):
            backend_id = str(params[0])
            limit = int(params[1]) if len(params) > 1 else None
            expired_keys: List[Tuple[str, str]] = [
                key
                for key, record in self.records.items()
                if key[0] == backend_id and _is_expired(record)
            ]
            if limit is not None:
                expired_keys = expired_keys[:limit]
            for key in expired_keys:
                self.records.pop(key, None)
            return FakeCursor(rowcount=len(expired_keys))

        return FakeCursor()

    def commit(self) -> None:
        """Record commits requested by the backend."""

        self.commit_count += 1


def _is_expired(record: Dict[str, Any]) -> bool:
    expires_at = record["expires_at"]
    return expires_at is not None and expires_at <= datetime.now(timezone.utc)


def test_postgres_cas_put_get_exists_delete_and_ttl() -> None:
    connection = FakePostgresConnection()
    backend = PostgresCAS(
        connection=connection,
        backend_id="postgres:test",
        default_ttl_seconds=60,
        create_table=False,
    )

    content_hash = backend.put(b"payload")

    assert backend.exists(content_hash)
    assert backend.get(content_hash) == b"payload"
    assert connection.records[(backend.backend_id, content_hash)]["expires_at"] is not None
    assert list(backend.iter_hashes()) == [content_hash]

    backend.delete(content_hash)

    assert not backend.exists(content_hash)


def test_postgres_cas_prunes_expired_objects() -> None:
    connection = FakePostgresConnection()
    backend = PostgresCAS(
        connection=connection,
        backend_id="postgres:test",
        create_table=False,
    )
    content_hash = backend.put(b"expired")
    connection.records[(backend.backend_id, content_hash)]["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    )

    assert not backend.exists(content_hash)
    with pytest.raises(KeyError):
        backend.get(content_hash)

    assert backend.prune_expired() == 1
    assert connection.records == {}


def test_migrate_cas_allows_postgres_alias_to_resolve_old_refs() -> None:
    source = InMemoryCAS(backend_id="filesystem:/old/cas")
    content_hash = source.put(b"migrated")
    connection = FakePostgresConnection()
    target = PostgresCAS(
        connection=connection,
        backend_id="postgres:agentstate",
        backend_aliases=[source.backend_id],
        create_table=False,
    )

    result = migrate_cas(source, target)
    ref = ContentRef(
        hash=content_hash,
        backend_id=source.backend_id,
        type_name="bytes",
        size_bytes=len(b"migrated"),
    )

    assert result.object_count == 1
    assert result.bytes_copied == len(b"migrated")
    assert ref.resolve(target) == b"migrated"
