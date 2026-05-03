"""Custom exceptions for agentref."""


class AgentRefError(Exception):
    """Base class for all agentref errors."""


class InlineSizeExceeded(AgentRefError):
    """Raised when an Inline field exceeds the configured size threshold."""


class AmbiguousFrameworkError(AgentRefError):
    """Raised when multiple supported frameworks appear active."""


class NoFrameworkDetectedError(AgentRefError):
    """Raised when no supported framework can be detected."""


class UnresolvedReferenceError(AgentRefError):
    """Raised when a content reference cannot be resolved from storage."""


class SerializationError(AgentRefError):
    """Raised when serialization or deserialization fails."""
