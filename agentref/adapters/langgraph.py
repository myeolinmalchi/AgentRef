"""LangGraph adapter for AgentRefState."""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Mapping as MappingABC
from functools import wraps
from typing import Annotated, Any, Callable, Dict, Mapping, Optional, Type, TypeVar, cast
from typing import TypedDict as _TypedDict

from agentref.adapters.base import BaseFrameworkAdapter, reducer_for_field
from agentref.core.state import AgentRefState

StateT = TypeVar("StateT", bound=AgentRefState)


class LangGraphAdapter(BaseFrameworkAdapter):
    """Adapter that produces LangGraph-compatible state schemas and checkpoints."""

    def wrap_state_class(
        self,
        state_cls: Optional[Type[AgentRefState]] = None,
    ) -> Any:
        """Return a TypedDict schema for ``StateGraph``.

        Externalized fields are checkpoint-safe ``ContentRef`` channels with a
        reference-aware reducer attached via ``Annotated``.
        """

        state_cls = self._require_state_cls(state_cls)
        annotations: Dict[str, Any] = {}
        for name, field in state_cls.fields().items():
            if field.kind == "externalized":
                annotations[name] = Annotated[Dict[str, Any], reducer_for_field(field)]
            else:
                annotations[name] = field.inner_type

        typed_dict_factory = cast(Any, _TypedDict)
        schema = typed_dict_factory(
            f"{state_cls.__name__}LangGraphState",
            annotations,
            total=False,
        )
        setattr(schema, "__agentref_origin__", state_cls)
        return schema

    def install_reducers(
        self,
        state_cls: Optional[Type[AgentRefState]] = None,
    ) -> Dict[str, Any]:
        """Return reducers for declared externalized channels."""

        state_cls = self._require_state_cls(state_cls)
        return {
            name: reducer_for_field(field)
            for name, field in state_cls.externalized_fields().items()
        }

    def externalize_node_update(
        self,
        state_cls_or_update: Any,
        update: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert a LangGraph node's partial update into checkpoint-safe values."""

        if update is None:
            state_cls = self._require_state_cls()
            raw_update = state_cls_or_update
        else:
            state_cls = self._require_state_cls(state_cls_or_update)
            raw_update = update
        if not isinstance(raw_update, MappingABC):
            raise TypeError(
                "LangGraph node updates must be mappings, found "
                f"{type(raw_update).__name__}."
            )

        converted = self.externalize_mapping(state_cls, raw_update)
        for name, field in state_cls.fields().items():
            if field.kind != "externalized" or name not in converted:
                continue
            ref = self._to_content_ref_if_reference_like(converted[name])
            if ref is not None:
                converted[name] = self._wrap_content_ref(ref)
        return converted

    def hydrate_state_for_node(
        self,
        state_cls_or_state: Any,
        state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Hydrate externalized fields before user node logic reads state."""

        if state is None:
            state_cls = self._require_state_cls()
            raw_state = state_cls_or_state
        else:
            state_cls = self._require_state_cls(state_cls_or_state)
            raw_state = state
        if not isinstance(raw_state, MappingABC):
            raise TypeError(
                "LangGraph state must be a mapping, found "
                f"{type(raw_state).__name__}."
            )
        return self.hydrate_mapping(state_cls, raw_state)

    def wrap_node(self, node: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a LangGraph node with hydrate-before and externalize-after logic."""

        state_cls = self._require_state_cls()

        if inspect.iscoroutinefunction(node):

            @wraps(node)
            async def async_wrapped(state: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
                hydrated = self.hydrate_mapping(state_cls, state)
                result = await node(hydrated, *args, **kwargs)
                return self._externalize_node_result(state_cls, result)

            return async_wrapped

        @wraps(node)
        def wrapped(state: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
            hydrated = self.hydrate_mapping(state_cls, state)
            result = node(hydrated, *args, **kwargs)
            return self._externalize_node_result(state_cls, result)

        return wrapped

    def node(self, node: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
        """Decorator alias for ``wrap_node``."""

        if node is None:

            def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                return self.wrap_node(fn)

            return decorator
        return self.wrap_node(node)

    def serialize_for_checkpoint(self, state_instance: Any) -> bytes:
        """Serialize a checkpoint-safe LangGraph state mapping."""

        if isinstance(state_instance, AgentRefState):
            return self._serialize_state_for_class(state_instance, type(state_instance))
        return pickle.dumps(
            self.checkpoint_dict_from_state(state_instance),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def deserialize_from_checkpoint(
        self,
        data: bytes,
        state_cls: Optional[Type[StateT]] = None,
    ) -> StateT:
        """Restore an AgentRefState instance from LangGraph checkpoint bytes."""

        resolved = cast(Type[StateT], self._require_state_cls(state_cls))
        return self._deserialize_state_for_class(data, resolved)

    def _externalize_node_result(
        self,
        state_cls: Type[AgentRefState],
        result: Any,
    ) -> Any:
        """Externalize mapping node results while leaving framework objects alone."""

        if isinstance(result, MappingABC):
            return self.externalize_node_update(state_cls, result)
        return result
