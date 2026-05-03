"""Content-addressed references with lazy hydration support."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, cast

from agentstate.exceptions import SerializationError, UnresolvedReferenceError

if TYPE_CHECKING:
    from agentstate.storage.base import BaseCASBackend


@dataclass(frozen=True)
class ContentRef:
    """Immutable reference to data stored in a content-addressed backend.

    Only this small reference is intended to appear in state checkpoints. The
    original object can be retrieved lazily with :meth:`resolve` while the
    underlying CAS entry is still present.
    """

    hash: str
    backend_id: str
    type_name: str
    size_bytes: int
    _resolved: Optional[Any] = field(
        default=None, init=False, repr=False, compare=False, hash=False
    )

    _STATE_KEYS: ClassVar[tuple[str, str, str, str]] = (
        "hash",
        "backend_id",
        "type_name",
        "size_bytes",
    )

    @property
    def is_resolved(self) -> bool:
        """Return whether this reference has hydrated data cached in memory."""

        return self._resolved is not None

    def __hash__(self) -> int:
        """Return a stable hash based only on the content hash."""

        return hash(self.hash)

    def __eq__(self, other: object) -> bool:
        """Compare references by content hash only."""

        if not isinstance(other, ContentRef):
            return NotImplemented
        return self.hash == other.hash

    def resolve(self, backend: Optional["BaseCASBackend"] = None) -> Any:
        """Hydrate and return the referenced object.

        Args:
            backend: Storage backend to read from. If omitted, this method tries
                to use ``agentstate.config.get_config().backend`` when that
                module exists in later phases.

        Raises:
            UnresolvedReferenceError: If no backend is available, the backend
                does not match this reference, or the object is missing.
        """

        if self._resolved is not None:
            return self._resolved

        active_backend = backend or self._configured_backend()
        if active_backend is None:
            raise UnresolvedReferenceError(
                f"Cannot resolve ContentRef {self.hash!r}: no backend was provided."
            )
        if not active_backend.can_resolve(self.backend_id):
            raise UnresolvedReferenceError(
                "Cannot resolve ContentRef "
                f"{self.hash!r}: backend {active_backend.backend_id!r} does not "
                f"match reference backend {self.backend_id!r}."
            )

        try:
            payload = active_backend.get(self.hash)
            value = active_backend.deserialize(payload, self.type_name)
        except KeyError as exc:
            raise UnresolvedReferenceError(
                f"Cannot resolve ContentRef {self.hash!r}: content is missing."
            ) from exc
        except Exception as exc:
            raise UnresolvedReferenceError(
                f"Cannot resolve ContentRef {self.hash!r}: {exc}"
            ) from exc

        object.__setattr__(self, "_resolved", value)
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON/msgpack-compatible primitive representation."""

        return {
            "hash": self.hash,
            "backend_id": self.backend_id,
            "type_name": self.type_name,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentRef":
        """Build a reference from a primitive mapping."""

        missing = [key for key in cls._STATE_KEYS if key not in data]
        if missing:
            raise SerializationError(
                f"ContentRef data is missing required key(s): {', '.join(missing)}"
            )
        return cls(
            hash=str(data["hash"]),
            backend_id=str(data["backend_id"]),
            type_name=str(data["type_name"]),
            size_bytes=int(data["size_bytes"]),
        )

    def to_json(self) -> str:
        """Serialize this reference to JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> "ContentRef":
        """Deserialize a reference from JSON."""

        try:
            loaded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise SerializationError("Invalid ContentRef JSON.") from exc
        if not isinstance(loaded, dict):
            raise SerializationError("ContentRef JSON must decode to an object.")
        return cls.from_dict(loaded)

    def to_msgpack(self) -> bytes:
        """Serialize this reference to msgpack bytes.

        Raises:
            SerializationError: If ``msgpack`` is not installed.
        """

        try:
            msgpack = cast(Any, importlib.import_module("msgpack"))
        except ModuleNotFoundError as exc:
            raise SerializationError("msgpack is required for msgpack encoding.") from exc
        return cast(bytes, msgpack.packb(self.to_dict(), use_bin_type=True))

    @classmethod
    def from_msgpack(cls, data: bytes) -> "ContentRef":
        """Deserialize a reference from msgpack bytes.

        Raises:
            SerializationError: If ``msgpack`` is unavailable or the payload is
                not a ContentRef object.
        """

        try:
            msgpack = cast(Any, importlib.import_module("msgpack"))
        except ModuleNotFoundError as exc:
            raise SerializationError("msgpack is required for msgpack decoding.") from exc
        try:
            loaded = msgpack.unpackb(data, raw=False)
        except Exception as exc:
            raise SerializationError("Invalid ContentRef msgpack payload.") from exc
        if not isinstance(loaded, dict):
            raise SerializationError("ContentRef msgpack must decode to an object.")
        return cls.from_dict(loaded)

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state without hydrated payload data."""

        return self.to_dict()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state without hydrated payload data."""

        restored = self.from_dict(state)
        object.__setattr__(self, "hash", restored.hash)
        object.__setattr__(self, "backend_id", restored.backend_id)
        object.__setattr__(self, "type_name", restored.type_name)
        object.__setattr__(self, "size_bytes", restored.size_bytes)
        object.__setattr__(self, "_resolved", None)

    @staticmethod
    def _configured_backend() -> Optional["BaseCASBackend"]:
        """Return the globally configured backend when available."""

        try:
            from agentstate.config import get_config
        except (ImportError, ModuleNotFoundError):
            return None

        config = get_config()
        return getattr(config, "backend", None)
