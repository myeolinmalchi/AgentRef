"""Generic marker types for AgentState fields."""

from __future__ import annotations

from typing import Any, Generic, Optional, Type, TypeVar, get_args, get_origin, overload

T = TypeVar("T")


class Inline(Generic[T]):
    """Marker for control-plane data stored directly in checkpoints."""

    @overload
    def __get__(
        self, instance: None, owner: Optional[Type[Any]] = None
    ) -> "Inline[T]":
        ...

    @overload
    def __get__(self, instance: object, owner: Optional[Type[Any]] = None) -> T:
        ...

    def __get__(self, instance: Optional[object], owner: Optional[Type[Any]] = None) -> Any:
        """Type-checker descriptor hook; runtime descriptors replace this."""

        raise AttributeError("Inline is a marker type and cannot be read directly.")

    def __set__(self, instance: object, value: T) -> None:
        """Type-checker descriptor hook; runtime descriptors replace this."""

        raise AttributeError("Inline is a marker type and cannot be assigned directly.")


class Externalized(Generic[T]):
    """Marker for data-plane values stored in CAS with checkpoint references."""

    @overload
    def __get__(
        self, instance: None, owner: Optional[Type[Any]] = None
    ) -> "Externalized[T]":
        ...

    @overload
    def __get__(self, instance: object, owner: Optional[Type[Any]] = None) -> T:
        ...

    def __get__(self, instance: Optional[object], owner: Optional[Type[Any]] = None) -> Any:
        """Type-checker descriptor hook; runtime descriptors replace this."""

        raise AttributeError(
            "Externalized is a marker type and cannot be read directly."
        )

    def __set__(self, instance: object, value: T) -> None:
        """Type-checker descriptor hook; runtime descriptors replace this."""

        raise AttributeError(
            "Externalized is a marker type and cannot be assigned directly."
        )


def is_inline_annotation(annotation: Any) -> bool:
    """Return whether ``annotation`` is ``Inline[T]``."""

    return get_origin(annotation) is Inline


def is_externalized_annotation(annotation: Any) -> bool:
    """Return whether ``annotation`` is ``Externalized[T]``."""

    return get_origin(annotation) is Externalized


def get_wrapped_type(annotation: Any) -> Any:
    """Return the inner ``T`` from ``Inline[T]`` or ``Externalized[T]``."""

    args = get_args(annotation)
    if not args:
        return Any
    return args[0]
