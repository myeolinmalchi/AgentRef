"""Custom exceptions for agentstate."""


class AgentStateError(Exception):
    """Base class for all agentstate errors."""


class InlineSizeExceeded(AgentStateError):
    """Raised when an Inline field exceeds the configured size threshold."""


class AmbiguousFrameworkError(AgentStateError):
    """Raised when multiple supported frameworks appear active."""


class NoFrameworkDetectedError(AgentStateError):
    """Raised when no supported framework can be detected."""


class UnresolvedReferenceError(AgentStateError):
    """Raised when a content reference cannot be resolved from storage."""


class SerializationError(AgentStateError):
    """Raised when serialization or deserialization fails."""
