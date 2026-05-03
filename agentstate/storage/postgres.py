"""PostgreSQL content-addressed storage backend."""

from __future__ import annotations

import importlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Optional, Sequence

from agentstate.exceptions import AgentStateError
from agentstate.storage.base import BaseCASBackend

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresCAS(BaseCASBackend):
    """PostgreSQL-backed CAS with optional TTL metadata."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        *,
        connection: Optional[Any] = None,
        table_name: str = "agentstate_cas_objects",
        backend_id: str = "postgres:agentstate",
        backend_aliases: Optional[Iterable[str]] = None,
        default_ttl_seconds: Optional[int] = None,
        update_last_accessed: bool = False,
        create_table: bool = True,
        commit: bool = True,
    ) -> None:
        """Create a PostgreSQL CAS backend.

        Args:
            dsn: PostgreSQL connection string. Required when ``connection`` is
                not supplied.
            connection: Existing DB-API/psycopg-style connection.
            table_name: Table name, optionally schema-qualified.
            backend_id: Identifier written to new ``ContentRef`` values.
            backend_aliases: Older backend ids this backend may resolve after
                payload migration.
            default_ttl_seconds: Optional TTL assigned to newly written objects.
            update_last_accessed: Whether reads update ``last_accessed_at``.
            create_table: Create the CAS table if it does not exist.
            commit: Commit writes when the connection exposes ``commit``.
        """

        if connection is None and not dsn:
            raise AgentStateError("PostgresCAS requires either dsn or connection.")
        if default_ttl_seconds is not None and default_ttl_seconds <= 0:
            raise AgentStateError("default_ttl_seconds must be positive.")

        self._connection = connection or self._connect(dsn)
        self._owns_connection = connection is None
        self._table_sql = _quote_qualified_name(table_name)
        self._backend_id = backend_id
        self._backend_aliases = frozenset(backend_aliases or ())
        self._default_ttl_seconds = default_ttl_seconds
        self._update_last_accessed = update_last_accessed
        self._commit_writes = commit

        if create_table:
            self._create_table()

    @property
    def backend_id(self) -> str:
        """Stable identifier for this backend instance."""

        return self._backend_id

    def can_resolve(self, backend_id: str) -> bool:
        """Return whether this backend can resolve refs for ``backend_id``."""

        return backend_id == self.backend_id or backend_id in self._backend_aliases

    def put(self, data: bytes) -> str:
        """Store bytes in PostgreSQL and return their SHA-256 hash."""

        content_hash = self.hash_bytes(data)
        expires_at = self._expires_at_for_new_object()
        sql = f"""
            INSERT INTO {self._table_sql} AS target
                (backend_id, hash, payload, size_bytes, expires_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (backend_id, hash) DO UPDATE
            SET expires_at = COALESCE(EXCLUDED.expires_at, target.expires_at)
        """
        self._execute(sql, (self.backend_id, content_hash, data, len(data), expires_at))
        self._commit()
        return content_hash

    def get(self, hash: str) -> bytes:
        """Return bytes for ``hash``.

        Raises:
            KeyError: If the hash is not present or has expired.
        """

        cursor = self._execute(
            f"""
            SELECT payload
            FROM {self._table_sql}
            WHERE backend_id = %s
              AND hash = %s
              AND (expires_at IS NULL OR expires_at > NOW())
            """,
            (self.backend_id, hash),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(hash)

        if self._update_last_accessed:
            self._execute(
                f"""
                UPDATE {self._table_sql}
                SET last_accessed_at = NOW()
                WHERE backend_id = %s AND hash = %s
                """,
                (self.backend_id, hash),
            )
            self._commit()

        return bytes(row[0])

    def exists(self, hash: str) -> bool:
        """Return whether ``hash`` exists and has not expired."""

        cursor = self._execute(
            f"""
            SELECT 1
            FROM {self._table_sql}
            WHERE backend_id = %s
              AND hash = %s
              AND (expires_at IS NULL OR expires_at > NOW())
            """,
            (self.backend_id, hash),
        )
        return cursor.fetchone() is not None

    def delete(self, hash: str) -> None:
        """Delete ``hash`` from storage if present."""

        self._execute(
            f"DELETE FROM {self._table_sql} WHERE backend_id = %s AND hash = %s",
            (self.backend_id, hash),
        )
        self._commit()

    def iter_hashes(self) -> Iterator[str]:
        """Yield non-expired hashes stored for this backend id."""

        cursor = self._execute(
            f"""
            SELECT hash
            FROM {self._table_sql}
            WHERE backend_id = %s
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY hash
            """,
            (self.backend_id,),
        )
        for row in cursor:
            yield str(row[0])

    def prune_expired(self, *, limit: Optional[int] = None) -> int:
        """Delete expired objects and return the database row count."""

        if limit is not None and limit <= 0:
            raise AgentStateError("limit must be positive.")

        if limit is None:
            cursor = self._execute(
                f"""
                DELETE FROM {self._table_sql}
                WHERE backend_id = %s
                  AND expires_at IS NOT NULL
                  AND expires_at <= NOW()
                """,
                (self.backend_id,),
            )
        else:
            cursor = self._execute(
                f"""
                WITH doomed AS (
                    SELECT hash
                    FROM {self._table_sql}
                    WHERE backend_id = %s
                      AND expires_at IS NOT NULL
                      AND expires_at <= NOW()
                    LIMIT %s
                )
                DELETE FROM {self._table_sql} AS target
                USING doomed
                WHERE target.backend_id = %s
                  AND target.hash = doomed.hash
                """,
                (self.backend_id, limit, self.backend_id),
            )
        self._commit()
        return int(getattr(cursor, "rowcount", 0) or 0)

    def close(self) -> None:
        """Close an owned connection."""

        if self._owns_connection and hasattr(self._connection, "close"):
            self._connection.close()

    def __enter__(self) -> "PostgresCAS":
        """Return this backend for context-manager use."""

        return self

    def __exit__(self, *_exc: object) -> None:
        """Close an owned connection on context-manager exit."""

        self.close()

    def _create_table(self) -> None:
        """Create the CAS table and TTL index when missing."""

        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_sql} (
                backend_id text NOT NULL,
                hash text NOT NULL,
                payload bytea NOT NULL,
                size_bytes bigint NOT NULL,
                created_at timestamptz NOT NULL DEFAULT NOW(),
                last_accessed_at timestamptz,
                expires_at timestamptz,
                PRIMARY KEY (backend_id, hash)
            )
            """
        )
        self._execute(
            f"""
            CREATE INDEX IF NOT EXISTS
                agentstate_cas_objects_expires_at_idx
            ON {self._table_sql} (expires_at)
            """
        )
        self._commit()

    def _expires_at_for_new_object(self) -> Optional[datetime]:
        """Return an expiry timestamp for new writes."""

        if self._default_ttl_seconds is None:
            return None
        return datetime.now(timezone.utc) + timedelta(seconds=self._default_ttl_seconds)

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        """Execute SQL on psycopg3 or DB-API style connections."""

        if hasattr(self._connection, "execute"):
            return self._connection.execute(sql, params)
        cursor = self._connection.cursor()
        cursor.execute(sql, params)
        return cursor

    def _commit(self) -> None:
        """Commit writes when configured to do so."""

        if self._commit_writes and hasattr(self._connection, "commit"):
            self._connection.commit()

    @staticmethod
    def _connect(dsn: Optional[str]) -> Any:
        """Open a psycopg connection."""

        try:
            psycopg = importlib.import_module("psycopg")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "PostgresCAS requires psycopg. Install with "
                "`pip install 'agentstate[postgres]'`."
            ) from exc
        return psycopg.connect(dsn)


def _quote_qualified_name(name: str) -> str:
    """Return a safely quoted SQL identifier or qualified identifier."""

    parts = name.split(".")
    if not parts or any(not _IDENTIFIER_RE.match(part) for part in parts):
        raise AgentStateError(f"Invalid PostgreSQL table name: {name!r}.")
    return ".".join(f'"{part}"' for part in parts)
