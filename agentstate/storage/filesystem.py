"""Filesystem content-addressed storage backend."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator, Union

from agentstate.storage.base import BaseCASBackend


class FilesystemCAS(BaseCASBackend):
    """Filesystem-backed CAS using a git-object-like directory layout."""

    def __init__(
        self,
        root: Union[str, Path],
        backend_id: str = "",
    ) -> None:
        """Create a filesystem CAS rooted at ``root``.

        Objects are stored at ``root/{hash[:2]}/{hash[2:]}``. Writes use a
        temporary file in the target shard directory followed by ``os.replace``.
        """

        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._backend_id = backend_id or f"filesystem:{self.root}"

    @property
    def backend_id(self) -> str:
        """Stable identifier for this backend instance."""

        return self._backend_id

    def put(self, data: bytes) -> str:
        """Store bytes atomically and return their SHA-256 hash."""

        content_hash = self.hash_bytes(data)
        path = self._path_for_hash(content_hash)
        if path.exists():
            return content_hash

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                delete=False,
            ) as temp_file:
                temp_name = temp_file.name
                temp_file.write(data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_name, path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
        return content_hash

    def get(self, hash: str) -> bytes:
        """Return bytes for ``hash``.

        Raises:
            KeyError: If the hash is not present.
        """

        path = self._path_for_hash(hash)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(hash) from exc

    def exists(self, hash: str) -> bool:
        """Return whether ``hash`` exists in storage."""

        return self._path_for_hash(hash).is_file()

    def delete(self, hash: str) -> None:
        """Delete ``hash`` from storage if present."""

        try:
            self._path_for_hash(hash).unlink()
        except FileNotFoundError:
            return

    def iter_hashes(self) -> Iterator[str]:
        """Yield content hashes currently present in storage."""

        for shard in sorted(self.root.iterdir()):
            if not shard.is_dir() or len(shard.name) != 2:
                continue
            for path in sorted(shard.iterdir()):
                if path.is_file() and not path.name.startswith("."):
                    yield f"{shard.name}{path.name}"

    def path_for_hash(self, hash: str) -> Path:
        """Return the object path for ``hash``."""

        return self._path_for_hash(hash)

    def _path_for_hash(self, hash: str) -> Path:
        """Return the sharded path for ``hash``."""

        if len(hash) < 3:
            raise KeyError(hash)
        return self.root / hash[:2] / hash[2:]
