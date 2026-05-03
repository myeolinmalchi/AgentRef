"""Base interfaces and helpers for framework adapters."""

from __future__ import annotations

import inspect
import pickle
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Dict, Iterator, Mapping, Optional, Type, TypeVar

from agentstate.config import get_config
from agentstate.core.invariants import validate_checkpoint_state
from agentstate.core.reducers import ref_aware_replace
from agentstate.core.reference import ContentRef
from agentstate.core.state import AgentState, StateField

StateT = TypeVar("StateT", bound=AgentState)


class BaseFrameworkAdapter(ABC):
    """Common interface for framework-specific adapters."""

    def __init__(self, state_cls: Optional[Type[AgentState]] = None) -> None:
        """Create an adapter optionally bound to one ``AgentState`` class."""

        if state_cls is not None and (
            not isinstance(state_cls, type) or not issubclass(state_cls, AgentState)
        ):
            raise TypeError(
                "state_cls must be an AgentState subclass, found "
                f"{type(state_cls).__name__}."
            )
        self._state_cls = state_cls

    @property
    def state_cls(self) -> Optional[Type[AgentState]]:
        """Return the state class bound to this adapter, if any."""

        return self._state_cls

    def _require_state_cls(
        self,
        state_cls: Optional[Type[AgentState]] = None,
    ) -> Type[AgentState]:
        """Return an explicit or constructor-bound state class."""

        selected = state_cls if state_cls is not None else self._state_cls
        if selected is None:
            raise TypeError(
                "A state class is required. Pass it to the adapter constructor "
                "or to this method."
            )
        if not isinstance(selected, type) or not issubclass(selected, AgentState):
            raise TypeError(
                "state_cls must be an AgentState subclass, found "
                f"{type(selected).__name__}."
            )
        return selected

    @abstractmethod
    def wrap_state_class(
        self,
        state_cls: Optional[Type[AgentState]] = None,
    ) -> Any:
        """Convert an ``AgentState`` class into a framework schema."""

    @abstractmethod
    def install_reducers(
        self,
        state_cls: Optional[Type[AgentState]] = None,
    ) -> Dict[str, Any]:
        """Return reducer functions for framework state channels."""

    @abstractmethod
    def serialize_for_checkpoint(self, state_instance: Any) -> bytes:
        """Serialize checkpoint-safe state bytes for a framework checkpointer."""

    @abstractmethod
    def deserialize_from_checkpoint(
        self,
        data: bytes,
        state_cls: Optional[Type[StateT]] = None,
    ) -> StateT:
        """Restore an ``AgentState`` instance from checkpoint bytes."""

    def schema(self) -> Any:
        """Return the framework schema for the constructor-bound state class."""

        return self.wrap_state_class(self._require_state_cls())

    def reducers(self) -> Dict[str, Any]:
        """Return reducers for the constructor-bound state class."""

        return self.install_reducers(self._require_state_cls())

    def externalize(
        self,
        values: Mapping[str, Any],
        state_cls: Optional[Type[AgentState]] = None,
    ) -> Dict[str, Any]:
        """Externalize a mapping using an explicit or constructor-bound state."""

        return self.externalize_mapping(self._require_state_cls(state_cls), values)

    def hydrate(
        self,
        values: Mapping[str, Any],
        state_cls: Optional[Type[AgentState]] = None,
    ) -> Dict[str, Any]:
        """Hydrate a mapping using an explicit or constructor-bound state."""

        return self.hydrate_mapping(self._require_state_cls(state_cls), values)

    def externalize_mapping(
        self,
        state_cls: Type[AgentState],
        values: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return ``values`` with externalized fields converted to ContentRef."""

        unknown = set(values) - set(state_cls.fields())
        if unknown:
            raise KeyError(
                f"Unknown field(s) for {state_cls.__name__}: "
                f"{', '.join(sorted(unknown))}"
            )

        result: Dict[str, Any] = {}
        for name, value in values.items():
            field = state_cls.fields()[name]
            if field.kind == "externalized":
                result[name] = self._to_content_ref(value)
            else:
                result[name] = value
        return result

    def hydrate_mapping(
        self,
        state_cls: Type[AgentState],
        values: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return ``values`` with ContentRef externalized fields hydrated."""

        result = dict(values)
        for name, field in state_cls.fields().items():
            if field.kind != "externalized" or name not in result:
                continue
            value = result[name]
            ref = self._to_content_ref_if_reference_like(value)
            if ref is not None:
                result[name] = ref.resolve(get_config().backend)
        return result

    def checkpoint_dict_from_state(self, state_instance: Any) -> Dict[str, Any]:
        """Extract a checkpoint-safe dict from state-like objects."""

        if isinstance(state_instance, AgentState):
            return state_instance.to_checkpoint_dict()
        if isinstance(state_instance, Mapping):
            return dict(state_instance)
        raise TypeError(
            "state_instance must be an AgentState or mapping, found "
            f"{type(state_instance).__name__}."
        )

    def _serialize_state_for_class(
        self,
        state_instance: Any,
        state_cls: Type[AgentState],
    ) -> bytes:
        """Serialize state after validating it against ``state_cls``."""

        checkpoint = self.checkpoint_dict_from_state(state_instance)
        validate_checkpoint_state(state_cls, checkpoint)
        return pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize_state_for_class(
        self, data: bytes, state_cls: Type[StateT]
    ) -> StateT:
        """Deserialize checkpoint bytes and restore ``state_cls``."""

        loaded = pickle.loads(data)
        if not isinstance(loaded, Mapping):
            raise TypeError("Checkpoint payload must deserialize to a mapping.")
        return state_cls.from_checkpoint_dict(
            self._normalize_reference_wrappers(state_cls, loaded)
        )

    def _to_content_ref(self, value: Any) -> ContentRef:
        """Store raw values in CAS and return a ``ContentRef``."""

        if isinstance(value, ContentRef):
            return value

        config = get_config()
        payload = config.backend.serialize(value)
        content_hash = config.backend.put(payload)
        return ContentRef(
            hash=content_hash,
            backend_id=config.backend.backend_id,
            type_name=type(value).__name__,
            size_bytes=len(payload),
        )

    def _wrap_content_ref(self, ref: ContentRef) -> Dict[str, Dict[str, Any]]:
        """Return a framework-serializer-safe reference wrapper."""

        return {"agentstate_ref": ref.to_dict()}

    def _to_content_ref_if_reference_like(self, value: Any) -> Optional[ContentRef]:
        """Return a ContentRef for supported reference representations."""

        if isinstance(value, ContentRef):
            return value
        if isinstance(value, Mapping):
            wrapper = value.get("agentstate_ref")
            if isinstance(wrapper, Mapping):
                return ContentRef.from_dict(dict(wrapper))
            required_keys = {"hash", "backend_id", "type_name", "size_bytes"}
            if required_keys.issubset(value):
                return ContentRef.from_dict(dict(value))
        return None

    def _normalize_reference_wrappers(
        self,
        state_cls: Type[AgentState],
        values: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Convert framework-safe wrappers back to ContentRef for AgentState."""

        normalized = dict(values)
        for name, field in state_cls.fields().items():
            if field.kind != "externalized" or name not in normalized:
                continue
            ref = self._to_content_ref_if_reference_like(normalized[name])
            if ref is not None:
                normalized[name] = ref
        return normalized


def reducer_for_field(field: StateField) -> Any:
    """Return the reducer that best matches an AgentState field."""

    if field.kind == "externalized":
        return ref_aware_replace
    return ref_aware_replace


class MappingStoreProxy(MutableMapping[str, Any]):
    """Mutable mapping proxy that externalizes writes and hydrates reads."""

    def __init__(
        self,
        adapter: BaseFrameworkAdapter,
        state_cls: Type[AgentState],
        store: MutableMapping[str, Any],
    ) -> None:
        """Create a proxy over a framework-owned mutable store."""

        self._adapter = adapter
        self._state_cls = state_cls
        self._store = store

    def __getitem__(self, key: str) -> Any:
        """Return a hydrated value for declared externalized fields."""

        value = self._store[key]
        field = self._state_cls.fields().get(key)
        if field is not None and field.kind == "externalized":
            ref = self._adapter._to_content_ref_if_reference_like(value)
            if ref is not None:
                return ref.resolve(get_config().backend)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a value, externalizing declared externalized fields."""

        self._store[key] = self._adapter.externalize_mapping(
            self._state_cls,
            {key: value},
        )[key]

    def __delitem__(self, key: str) -> None:
        """Delete ``key`` from the underlying store."""

        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over underlying keys."""

        return iter(self._store)

    def __len__(self) -> int:
        """Return underlying store length."""

        return len(self._store)

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Return a checkpoint-safe copy of the underlying store."""

        return dict(self._store)


class AsyncContextStoreProxy:
    """Async store proxy for framework context stores with ``get``/``set`` APIs."""

    def __init__(
        self,
        adapter: BaseFrameworkAdapter,
        state_cls: Type[AgentState],
        store: Any,
        *,
        wrap_externalized_refs: bool = False,
    ) -> None:
        """Create an async proxy over a context store."""

        self._adapter = adapter
        self._state_cls = state_cls
        self._store = store
        self._wrap_externalized_refs = wrap_externalized_refs

    async def get(self, key: str, default: Any = None) -> Any:
        """Return a hydrated value from the wrapped store."""

        value = await self._get_raw(key, default)
        field = self._state_cls.fields().get(key)
        if field is not None and field.kind == "externalized":
            ref = self._adapter._to_content_ref_if_reference_like(value)
            if ref is not None:
                return ref.resolve(get_config().backend)
        return value

    async def set(self, key: str, value: Any) -> None:
        """Set a value, externalizing declared externalized fields."""

        converted = self._adapter.externalize_mapping(self._state_cls, {key: value})[
            key
        ]
        field = self._state_cls.fields().get(key)
        if (
            self._wrap_externalized_refs
            and field is not None
            and field.kind == "externalized"
        ):
            ref = self._adapter._to_content_ref_if_reference_like(converted)
            if ref is not None:
                converted = self._adapter._wrap_content_ref(ref)
        await self._call_store("set", key, converted)

    async def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Return raw store values for declared fields."""

        result: Dict[str, Any] = {}
        missing = object()
        for name in self._state_cls.fields():
            value = await self._get_raw(name, missing)
            if value is not missing:
                result[name] = value
        return result

    async def _get_raw(self, key: str, default: Any) -> Any:
        """Return raw value from the wrapped store."""

        return await self._call_store("get", key, default)

    async def _call_store(self, method_name: str, *args: Any) -> Any:
        """Call a possibly-async store method."""

        method = getattr(self._store, method_name)
        result = method(*args)
        if inspect.isawaitable(result):
            return await result
        return result
