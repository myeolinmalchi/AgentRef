"""Runtime invariant checks for AgentRefState instances and checkpoints."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Type

from agentref.config import get_config
from agentref.core.reference import ContentRef
from agentref.core.state import AgentRefState
from agentref.exceptions import AgentRefError, InlineSizeExceeded
from agentref.storage.base import BaseCASBackend


def validate_inline_value(
    field_name: str,
    value: Any,
    *,
    threshold_bytes: Optional[int] = None,
    backend: Optional[BaseCASBackend] = None,
) -> None:
    """Validate that an inline value fits within the configured size limit.

    Raises:
        InlineSizeExceeded: If serialized bytes exceed the configured limit.
    """

    config = get_config()
    active_backend = backend or config.backend
    active_threshold = (
        config.inline_threshold_bytes
        if threshold_bytes is None
        else threshold_bytes
    )
    payload = active_backend.serialize(value)
    size_bytes = len(payload)
    if size_bytes > active_threshold:
        raise InlineSizeExceeded(
            f"Inline field {field_name!r} serialized to {size_bytes} bytes, "
            f"exceeding the configured limit of {active_threshold} bytes."
        )


def validate_externalized_ref(
    field_name: str,
    value: Any,
    *,
    backend: Optional[BaseCASBackend] = None,
    require_exists: bool = False,
) -> None:
    """Validate that an externalized checkpoint value is a ``ContentRef``.

    Args:
        field_name: Field being checked, used in error messages.
        value: Checkpoint value for that field.
        backend: Backend expected to contain the referenced payload.
        require_exists: When true, assert the referenced hash exists.

    Raises:
        AgentRefError: If ``value`` is not a valid reference or storage is
            inconsistent.
    """

    ref = _coerce_ref_like(value)
    if ref is None:
        raise AgentRefError(
            f"Externalized field {field_name!r} must store ContentRef or a "
            f"ContentRef wrapper in checkpoint state, found {type(value).__name__}."
        )

    active_backend = backend or get_config().backend
    if not active_backend.can_resolve(ref.backend_id):
        raise AgentRefError(
            f"Externalized field {field_name!r} references backend "
            f"{ref.backend_id!r}, but active backend is "
            f"{active_backend.backend_id!r}."
        )

    if require_exists and not active_backend.exists(ref.hash):
        raise AgentRefError(
            f"Externalized field {field_name!r} references missing content "
            f"hash {ref.hash!r}."
        )


def validate_checkpoint_state(
    state_cls: Type[AgentRefState],
    checkpoint: Mapping[str, Any],
    *,
    backend: Optional[BaseCASBackend] = None,
    require_externalized_exists: bool = False,
) -> None:
    """Validate checkpoint values against an ``AgentRefState`` class.

    This check enforces the core checkpoint invariant: every declared
    ``Externalized`` field present in the mapping must be represented by
    ``ContentRef`` only.
    """

    unknown = set(checkpoint) - set(state_cls.fields())
    if unknown:
        raise AgentRefError(
            f"Checkpoint for {state_cls.__name__} contains unknown field(s): "
            f"{', '.join(sorted(unknown))}"
        )

    for name, field in state_cls.fields().items():
        if name not in checkpoint:
            continue

        value = checkpoint[name]
        if field.kind == "externalized":
            validate_externalized_ref(
                name,
                value,
                backend=backend,
                require_exists=require_externalized_exists,
            )
        elif field.kind == "inline":
            validate_inline_value(name, value, backend=backend)


def validate_agent_ref(
    state: AgentRefState,
    *,
    backend: Optional[BaseCASBackend] = None,
    require_externalized_exists: bool = False,
) -> None:
    """Validate an ``AgentRefState`` instance's checkpoint-safe representation."""

    validate_checkpoint_state(
        type(state),
        state.to_checkpoint_dict(),
        backend=backend,
        require_externalized_exists=require_externalized_exists,
    )


def _coerce_ref_like(value: Any) -> Optional[ContentRef]:
    """Return a ContentRef from supported checkpoint-safe representations."""

    if isinstance(value, ContentRef):
        return value
    if isinstance(value, Mapping):
        wrapper = value.get("agentref_ref")
        if isinstance(wrapper, Mapping):
            return ContentRef.from_dict(dict(wrapper))
        required_keys = {"hash", "backend_id", "type_name", "size_bytes"}
        if required_keys.issubset(value):
            return ContentRef.from_dict(dict(value))
    return None
