"""LlamaIndex Workflow adapter for AgentRefState."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar, cast

from agentref.adapters.base import (
    AsyncContextStoreProxy,
    BaseFrameworkAdapter,
    MappingStoreProxy,
    reducer_for_field,
)
from agentref.core.state import AgentRefState

StateT = TypeVar("StateT", bound=AgentRefState)


@dataclass(frozen=True)
class LlamaIndexStateSpec:
    """Description of an AgentRefState class for LlamaIndex Context stores."""

    state_cls: Type[AgentRefState]
    fields: Dict[str, str]


class LlamaIndexContextStoreProxy(MappingStoreProxy):
    """Proxy for a LlamaIndex ``Context.store``-like mapping."""


class LlamaIndexAsyncContextStoreProxy(AsyncContextStoreProxy):
    """Proxy for real LlamaIndex Workflow ``Context.store`` objects."""


class LlamaIndexAdapter(BaseFrameworkAdapter):
    """Adapter for LlamaIndex Workflow Context store dictionaries."""

    def wrap_state_class(
        self,
        state_cls: Optional[Type[AgentRefState]] = None,
    ) -> LlamaIndexStateSpec:
        """Return metadata describing Context store fields."""

        state_cls = self._require_state_cls(state_cls)
        return LlamaIndexStateSpec(
            state_cls=state_cls,
            fields={name: field.kind for name, field in state_cls.fields().items()},
        )

    def install_reducers(
        self,
        state_cls: Optional[Type[AgentRefState]] = None,
    ) -> Dict[str, Any]:
        """Return reducers for externalized Context store fields."""

        state_cls = self._require_state_cls(state_cls)
        return {
            name: reducer_for_field(field)
            for name, field in state_cls.externalized_fields().items()
        }

    def context_store(
        self,
        state_cls_or_store: Any,
        store: Optional[Any] = None,
    ) -> Any:
        """Wrap a Context.store-like mapping with AgentRefState semantics."""

        if store is None:
            state_cls = self._require_state_cls()
            raw_store = state_cls_or_store
        else:
            state_cls = self._require_state_cls(state_cls_or_store)
            raw_store = store

        if not isinstance(raw_store, MutableMapping) and all(
            hasattr(raw_store, method_name) for method_name in ("get", "set")
        ):
            return LlamaIndexAsyncContextStoreProxy(
                self,
                state_cls,
                raw_store,
                wrap_externalized_refs=True,
            )
        return LlamaIndexContextStoreProxy(self, state_cls, raw_store)

    def serialize_for_checkpoint(self, state_instance: Any) -> bytes:
        """Serialize a LlamaIndex Context-compatible checkpoint."""

        if isinstance(state_instance, AgentRefState):
            return self._serialize_state_for_class(state_instance, type(state_instance))
        return self._serialize_state_for_class_like_mapping(state_instance)

    def deserialize_from_checkpoint(
        self,
        data: bytes,
        state_cls: Optional[Type[StateT]] = None,
    ) -> StateT:
        """Restore an AgentRefState instance from Context checkpoint bytes."""

        resolved = cast(Type[StateT], self._require_state_cls(state_cls))
        return self._deserialize_state_for_class(data, resolved)

    def _serialize_state_for_class_like_mapping(self, state_instance: Any) -> bytes:
        """Serialize a mapping without an explicit state class."""

        import pickle

        return pickle.dumps(
            self.checkpoint_dict_from_state(state_instance),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
