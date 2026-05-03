"""Abstract content-addressed storage backend."""

from __future__ import annotations

import hashlib
import importlib
import pickle
from abc import ABC, abstractmethod
from typing import Any, cast

from agentstate.exceptions import SerializationError


class BaseCASBackend(ABC):
    """Interface for content-addressed storage backends."""

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Stable identifier for this backend instance."""

    @abstractmethod
    def put(self, data: bytes) -> str:
        """Store bytes and return their content hash.

        Repeated writes of identical bytes must return the same hash and should
        not duplicate storage.
        """

    @abstractmethod
    def get(self, hash: str) -> bytes:
        """Return bytes for ``hash``.

        Raises:
            KeyError: If the hash does not exist.
        """

    @abstractmethod
    def exists(self, hash: str) -> bool:
        """Return whether ``hash`` exists in storage."""

    @abstractmethod
    def delete(self, hash: str) -> None:
        """Delete ``hash`` from storage if present."""

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Return the SHA-256 content hash for ``data``."""

        return hashlib.sha256(data).hexdigest()

    def serialize(self, obj: Any) -> bytes:
        """Serialize a Python object to bytes.

        ``bytes`` values are passed through unchanged. Other values use
        ``msgpack`` when installed and fall back to ``pickle`` otherwise.
        """

        if isinstance(obj, bytes):
            return obj

        try:
            msgpack = cast(Any, importlib.import_module("msgpack"))
        except ModuleNotFoundError:
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            return cast(
                bytes,
                msgpack.packb(obj, use_bin_type=True, strict_types=False),
            )
        except Exception as exc:
            try:
                return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as pickle_exc:
                raise SerializationError(
                    f"Object of type {type(obj).__name__!r} cannot be serialized."
                ) from pickle_exc or exc

    def deserialize(self, data: bytes, type_name: str) -> Any:
        """Deserialize bytes into a Python object.

        Args:
            data: Serialized bytes.
            type_name: Original type name. ``bytes`` uses pass-through behavior.
        """

        if type_name == "bytes":
            return data

        try:
            msgpack = cast(Any, importlib.import_module("msgpack"))
        except ModuleNotFoundError:
            pass
        else:
            try:
                return msgpack.unpackb(data, raw=False)
            except Exception:
                pass

        try:
            return pickle.loads(data)
        except Exception as exc:
            raise SerializationError(
                f"Payload cannot be deserialized as original type {type_name!r}."
            ) from exc
