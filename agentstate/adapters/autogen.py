"""AutoGen adapter for AgentState.

AutoGen versions expose less uniform state machinery than LangGraph and
LlamaIndex. This adapter therefore focuses on declared ``AgentState`` mappings
and explicit message-history externalization helpers instead of monkeypatching
framework Agent classes.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, TypeVar, cast

from agentstate.config import get_config
from agentstate.adapters.base import BaseFrameworkAdapter, reducer_for_field
from agentstate.core.reference import ContentRef
from agentstate.core.state import AgentState

StateT = TypeVar("StateT", bound=AgentState)


class AutoGenAdapter(BaseFrameworkAdapter):
    """Adapter for AutoGen state dictionaries and message histories."""

    def wrap_state_class(
        self,
        state_cls: Optional[Type[AgentState]] = None,
    ) -> Type[AgentState]:
        """Return ``state_cls`` because AutoGen has no single schema protocol."""

        return self._require_state_cls(state_cls)

    def install_reducers(
        self,
        state_cls: Optional[Type[AgentState]] = None,
    ) -> Dict[str, Any]:
        """Return reducers for declared externalized state fields."""

        state_cls = self._require_state_cls(state_cls)
        return {
            name: reducer_for_field(field)
            for name, field in state_cls.externalized_fields().items()
        }

    def externalize_state(
        self,
        state_cls_or_state: Any,
        state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert an AutoGen state mapping into checkpoint-safe values."""

        if state is None:
            state_cls = self._require_state_cls()
            raw_state = state_cls_or_state
        else:
            state_cls = self._require_state_cls(state_cls_or_state)
            raw_state = state
        return self.externalize_mapping(state_cls, raw_state)

    def hydrate_state(
        self,
        state_cls_or_state: Any,
        state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Hydrate an AutoGen state mapping for user code."""

        if state is None:
            state_cls = self._require_state_cls()
            raw_state = state_cls_or_state
        else:
            state_cls = self._require_state_cls(state_cls_or_state)
            raw_state = state
        return self.hydrate_mapping(state_cls, raw_state)

    def externalize_message_history(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        threshold_bytes: int = 0,
        keys: Iterable[str] = ("content", "result", "tool_result"),
    ) -> List[Dict[str, Any]]:
        """Externalize large message fields in AutoGen conversation history.

        Values whose serialized size is greater than ``threshold_bytes`` are
        replaced by a small ContentRef dictionary so message-history checkpoints
        do not store repeated large payloads.
        """

        key_set = set(keys)
        converted: List[Dict[str, Any]] = []
        for message in messages:
            next_message = dict(message)
            for key in key_set:
                if key not in next_message:
                    continue
                value = next_message[key]
                payload = get_config().backend.serialize(value)
                if len(payload) <= threshold_bytes:
                    continue
                ref = self._to_content_ref(value)
                next_message[key] = {"agentstate_ref": ref.to_dict()}
            converted.append(next_message)
        return converted

    def hydrate_message_history(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        keys: Iterable[str] = ("content", "result", "tool_result"),
    ) -> List[Dict[str, Any]]:
        """Hydrate message fields previously externalized by this adapter."""

        key_set = set(keys)
        hydrated: List[Dict[str, Any]] = []
        for message in messages:
            next_message = dict(message)
            for key in key_set:
                value = next_message.get(key)
                if not _is_ref_wrapper(value):
                    continue
                wrapper = cast(Dict[str, Dict[str, Any]], value)
                ref = ContentRef.from_dict(wrapper["agentstate_ref"])
                next_message[key] = ref.resolve(get_config().backend)
            hydrated.append(next_message)
        return hydrated

    def serialize_for_checkpoint(self, state_instance: Any) -> bytes:
        """Serialize an AutoGen-compatible state mapping."""

        if isinstance(state_instance, AgentState):
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
        """Restore an AgentState instance from AutoGen checkpoint bytes."""

        resolved = cast(Type[StateT], self._require_state_cls(state_cls))
        return self._deserialize_state_for_class(data, resolved)


def _is_ref_wrapper(value: Any) -> bool:
    """Return whether ``value`` is an adapter-created reference wrapper."""

    return isinstance(value, dict) and isinstance(value.get("agentstate_ref"), dict)
