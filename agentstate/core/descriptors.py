"""Descriptors that enforce Inline and Externalized field semantics."""

from __future__ import annotations

from typing import Any, Optional, Type

from agentstate.config import get_config
from agentstate.core.reference import ContentRef
from agentstate.exceptions import AgentStateError, InlineSizeExceeded


def type_name_for(annotation: Any, value: Optional[Any] = None) -> str:
    """Return a readable type name for a field annotation or runtime value."""

    source = annotation
    if value is not None:
        source = type(value)

    origin = getattr(source, "__origin__", None)
    if origin is not None:
        source = origin

    return getattr(source, "__name__", str(source))


def runtime_for(instance: Any) -> Any:
    """Return the runtime attached to ``instance`` or the global fallback."""

    return getattr(instance, "_agentstate_runtime", None) or get_config()


class InlineDescriptor:
    """Descriptor for ``Inline[T]`` fields.

    Values are stored directly in the owning ``AgentState`` instance after their
    serialized size is checked against the active configuration threshold.
    """

    def __init__(self, name: str, inner_type: Any) -> None:
        """Create a descriptor for ``name`` with declared ``inner_type``."""

        self.name = name
        self.inner_type = inner_type

    def __get__(self, instance: Any, owner: Optional[Type[Any]] = None) -> Any:
        """Return the stored inline value."""

        if instance is None:
            return self
        try:
            return instance._data[self.name]
        except KeyError as exc:
            raise AttributeError(
                f"Inline field {self.name!r} has not been assigned."
            ) from exc

    def __set__(self, instance: Any, value: Any) -> None:
        """Validate and store an inline value."""

        runtime = runtime_for(instance)
        payload = runtime.backend.serialize(value)
        size_bytes = len(payload)
        if size_bytes > runtime.inline_threshold_bytes:
            raise InlineSizeExceeded(
                f"Inline field {self.name!r} serialized to {size_bytes} bytes, "
                f"exceeding the configured limit of "
                f"{runtime.inline_threshold_bytes} bytes. Declare this field as "
                "Externalized[...] or reduce the value size."
            )
        instance._data[self.name] = value


class ExternalizedDescriptor:
    """Descriptor for ``Externalized[T]`` fields.

    Assigned values are serialized into the configured CAS backend immediately;
    only a ``ContentRef`` is retained in the owning state instance.
    """

    def __init__(self, name: str, inner_type: Any) -> None:
        """Create a descriptor for ``name`` with declared ``inner_type``."""

        self.name = name
        self.inner_type = inner_type

    def __get__(self, instance: Any, owner: Optional[Type[Any]] = None) -> Any:
        """Hydrate and return the original externalized value."""

        if instance is None:
            return self

        ref = self.get_ref(instance)
        return ref.resolve(runtime_for(instance).backend)

    def __set__(self, instance: Any, value: Any) -> None:
        """Store ``value`` in CAS and retain only its ``ContentRef``."""

        if isinstance(value, ContentRef):
            instance._data[self.name] = value
            return

        runtime = runtime_for(instance)
        payload = runtime.backend.serialize(value)
        content_hash = runtime.backend.put(payload)
        instance._data[self.name] = ContentRef(
            hash=content_hash,
            backend_id=runtime.backend.backend_id,
            type_name=type_name_for(self.inner_type, value),
            size_bytes=len(payload),
        )

    def get_ref(self, instance: Any) -> ContentRef:
        """Return the checkpoint-safe reference stored for this field."""

        try:
            ref = instance._data[self.name]
        except KeyError as exc:
            raise AttributeError(
                f"Externalized field {self.name!r} has not been assigned."
            ) from exc

        if not isinstance(ref, ContentRef):
            raise AgentStateError(
                f"Externalized field {self.name!r} must store ContentRef, "
                f"found {type(ref).__name__}."
            )
        return ref
