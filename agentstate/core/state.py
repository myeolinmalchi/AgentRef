"""AgentState base class and metaclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Mapping, Type, TypeVar, get_type_hints

from agentstate.core.descriptors import ExternalizedDescriptor, InlineDescriptor
from agentstate.core.reference import ContentRef
from agentstate.core.types import (
    get_wrapped_type,
    is_externalized_annotation,
    is_inline_annotation,
)
from agentstate.exceptions import AgentStateError

StateT = TypeVar("StateT", bound="AgentState")


@dataclass(frozen=True)
class StateField:
    """Metadata for one AgentState field."""

    name: str
    kind: str
    inner_type: Any
    annotation: Any


class AgentStateMeta(type):
    """Metaclass that installs descriptors from Inline/Externalized annotations."""

    def __new__(
        mcs: Type["AgentStateMeta"],
        name: str,
        bases: tuple[type, ...],
        namespace: Dict[str, Any],
    ) -> "AgentStateMeta":
        """Create a state class and attach field descriptors."""

        cls = super().__new__(mcs, name, bases, namespace)

        fields: Dict[str, StateField] = {}
        for base in bases:
            fields.update(getattr(base, "__agentstate_fields__", {}))

        own_annotations = namespace.get("__annotations__", {})
        resolved_hints = mcs._resolved_type_hints(cls)
        for field_name in own_annotations:
            if field_name.startswith("_"):
                continue

            annotation = resolved_hints.get(field_name, own_annotations[field_name])
            if is_inline_annotation(annotation):
                inner_type = get_wrapped_type(annotation)
                setattr(cls, field_name, InlineDescriptor(field_name, inner_type))
                fields[field_name] = StateField(
                    name=field_name,
                    kind="inline",
                    inner_type=inner_type,
                    annotation=annotation,
                )
            elif is_externalized_annotation(annotation):
                inner_type = get_wrapped_type(annotation)
                setattr(cls, field_name, ExternalizedDescriptor(field_name, inner_type))
                fields[field_name] = StateField(
                    name=field_name,
                    kind="externalized",
                    inner_type=inner_type,
                    annotation=annotation,
                )

        setattr(cls, "__agentstate_fields__", fields)
        return cls

    @staticmethod
    def _resolved_type_hints(cls: type) -> Dict[str, Any]:
        """Return resolved type hints for ``cls`` when possible."""

        try:
            return get_type_hints(cls)
        except Exception:
            return dict(getattr(cls, "__annotations__", {}))


class AgentState(metaclass=AgentStateMeta):
    """Framework-agnostic state object with safe checkpoint representation."""

    __agentstate_fields__: ClassVar[Dict[str, StateField]]
    _data: Dict[str, Any]

    def __init__(self, **values: Any) -> None:
        """Initialize state fields from keyword arguments."""

        self._data = {}
        self._assign_initial_values(values)

    @classmethod
    def fields(cls) -> Mapping[str, StateField]:
        """Return declared AgentState field metadata."""

        return dict(cls.__agentstate_fields__)

    @classmethod
    def inline_fields(cls) -> Mapping[str, StateField]:
        """Return metadata for fields declared as ``Inline``."""

        return {
            name: field
            for name, field in cls.__agentstate_fields__.items()
            if field.kind == "inline"
        }

    @classmethod
    def externalized_fields(cls) -> Mapping[str, StateField]:
        """Return metadata for fields declared as ``Externalized``."""

        return {
            name: field
            for name, field in cls.__agentstate_fields__.items()
            if field.kind == "externalized"
        }

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Return a checkpoint-safe mapping.

        Externalized fields are represented by ``ContentRef`` objects only; any
        hydrated payload cached inside a reference is intentionally excluded by
        ``ContentRef`` serialization methods.
        """

        return dict(self._data)

    def to_hydrated_dict(self) -> Dict[str, Any]:
        """Return all field values, hydrating externalized data."""

        return {name: getattr(self, name) for name in self.__agentstate_fields__}

    def to_langgraph_state(self) -> Dict[str, Any]:
        """Return a LangGraph-compatible checkpoint-safe state mapping."""

        return self.to_checkpoint_dict()

    def to_llamaindex_context_dict(self) -> Dict[str, Any]:
        """Return a LlamaIndex Context-compatible checkpoint-safe mapping."""

        return self.to_checkpoint_dict()

    def to_autogen_state(self) -> Dict[str, Any]:
        """Return an AutoGen-compatible checkpoint-safe state mapping."""

        return self.to_checkpoint_dict()

    def dispatch_to(self, framework: Any) -> Dict[str, Any]:
        """Dispatch this state to a framework-specific checkpoint mapping."""

        from agentstate.detection.framework import Framework, detect_active_framework

        selected = detect_active_framework(framework)
        if selected is Framework.LANGGRAPH:
            return self.to_langgraph_state()
        if selected is Framework.LLAMAINDEX:
            return self.to_llamaindex_context_dict()
        if selected is Framework.AUTOGEN:
            return self.to_autogen_state()
        raise AgentStateError(f"Unsupported framework: {selected!r}")

    @classmethod
    def from_checkpoint_dict(cls: Type[StateT], state: Mapping[str, Any]) -> StateT:
        """Restore an instance from a checkpoint-safe mapping."""

        instance = cls()
        unknown = set(state) - set(cls.__agentstate_fields__)
        if unknown:
            raise AgentStateError(
                f"Unknown field(s) for {cls.__name__}: {', '.join(sorted(unknown))}"
            )

        for name, value in state.items():
            field = cls.__agentstate_fields__[name]
            if field.kind == "externalized":
                setattr(instance, name, cls._coerce_content_ref(name, value))
            else:
                setattr(instance, name, value)
        return instance

    @classmethod
    def from_langgraph_state(cls: Type[StateT], state: Mapping[str, Any]) -> StateT:
        """Restore from a LangGraph state mapping."""

        return cls.from_checkpoint_dict(state)

    @classmethod
    def from_llamaindex_context_dict(
        cls: Type[StateT], state: Mapping[str, Any]
    ) -> StateT:
        """Restore from a LlamaIndex Context state mapping."""

        return cls.from_checkpoint_dict(state)

    @classmethod
    def from_autogen_state(cls: Type[StateT], state: Mapping[str, Any]) -> StateT:
        """Restore from an AutoGen state mapping."""

        return cls.from_checkpoint_dict(state)

    def __getitem__(self, name: str) -> Any:
        """Return a hydrated field value by name."""

        self._validate_field_name(name)
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        """Assign a field value by name."""

        self._validate_field_name(name)
        setattr(self, name, value)

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is a declared and assigned state field."""

        return isinstance(name, str) and name in self._data

    def __repr__(self) -> str:
        """Return a debug representation without hydrating externalized fields."""

        fields = ", ".join(f"{name}={value!r}" for name, value in self._data.items())
        return f"{type(self).__name__}({fields})"

    def _assign_initial_values(self, values: Mapping[str, Any]) -> None:
        """Assign constructor values after validating field names."""

        unknown = set(values) - set(self.__agentstate_fields__)
        if unknown:
            raise AgentStateError(
                f"Unknown field(s) for {type(self).__name__}: "
                f"{', '.join(sorted(unknown))}"
            )

        for name, value in values.items():
            setattr(self, name, value)

    def _validate_field_name(self, name: str) -> None:
        """Raise if ``name`` is not a declared AgentState field."""

        if name not in self.__agentstate_fields__:
            raise KeyError(f"Unknown field for {type(self).__name__}: {name!r}")

    @staticmethod
    def _coerce_content_ref(name: str, value: Any) -> ContentRef:
        """Convert supported checkpoint reference representations to ContentRef."""

        if isinstance(value, ContentRef):
            return value
        if isinstance(value, dict):
            wrapper = value.get("agentstate_ref")
            if isinstance(wrapper, dict):
                return ContentRef.from_dict(wrapper)
            return ContentRef.from_dict(value)
        raise AgentStateError(
            f"Externalized field {name!r} must be restored from ContentRef "
            f"or ContentRef dict, found {type(value).__name__}."
        )
