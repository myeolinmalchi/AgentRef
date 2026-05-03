"""Tests for Phase 3 reducers and runtime invariants."""

from __future__ import annotations

import pytest

from agentref.config import configure
from agentref.core.invariants import (
    validate_agent_ref,
    validate_checkpoint_state,
    validate_externalized_ref,
    validate_inline_value,
)
from agentref.core.reducers import (
    ref_aware_dict_merge,
    ref_aware_list_append,
    ref_aware_replace,
)
from agentref.core.reference import ContentRef
from agentref.core.state import AgentRefState
from agentref.core.types import Externalized, Inline
from agentref.exceptions import AgentRefError, InlineSizeExceeded
from agentref.storage import InMemoryCAS


def _ref(content_hash: str, backend_id: str = "memory") -> ContentRef:
    return ContentRef(
        hash=content_hash,
        backend_id=backend_id,
        type_name="bytes",
        size_bytes=10,
    )


def test_ref_aware_dict_merge_replaces_without_hydrating_refs() -> None:
    left_ref = _ref("a" * 64)
    right_ref = _ref("b" * 64)
    left = {"docs": left_ref, "count": 1}
    right = {"docs": right_ref}

    merged = ref_aware_dict_merge(left, right)

    assert merged == {"docs": right_ref, "count": 1}
    assert not left_ref.is_resolved
    assert not right_ref.is_resolved


def test_ref_aware_dict_merge_keeps_equal_ref_without_extra_work() -> None:
    ref = _ref("a" * 64)
    merged = ref_aware_dict_merge({"docs": ref}, {"docs": _ref("a" * 64)})

    assert merged["docs"] == ref
    assert merged["docs"] is ref


def test_ref_aware_list_append_deduplicates_refs_by_hash_only() -> None:
    first = _ref("a" * 64, backend_id="memory-a")
    duplicate = _ref("a" * 64, backend_id="memory-b")
    second = _ref("b" * 64)

    merged = ref_aware_list_append([first, "raw"], [duplicate, "raw", second])

    assert merged == [first, "raw", "raw", second]
    assert not first.is_resolved
    assert not duplicate.is_resolved
    assert not second.is_resolved


def test_ref_aware_replace_returns_right_value() -> None:
    assert ref_aware_replace("old", "new") == "new"


def test_validate_inline_value_raises_when_serialized_size_exceeds_limit() -> None:
    configure(inline_threshold_bytes=3)

    with pytest.raises(InlineSizeExceeded, match="payload"):
        validate_inline_value("payload", b"abcd")


def test_validate_externalized_ref_rejects_raw_payload() -> None:
    with pytest.raises(AgentRefError, match="ContentRef"):
        validate_externalized_ref("docs", ["raw-doc"])


def test_validate_externalized_ref_checks_backend_id_and_existence() -> None:
    backend = InMemoryCAS("active")
    configure(backend=backend)
    wrong_backend_ref = _ref("a" * 64, backend_id="other")

    with pytest.raises(AgentRefError, match="references backend"):
        validate_externalized_ref("docs", wrong_backend_ref)

    missing_ref = _ref("b" * 64, backend_id="active")
    with pytest.raises(AgentRefError, match="missing content"):
        validate_externalized_ref("docs", missing_ref, require_exists=True)


def test_validate_agent_ref_accepts_safe_checkpoint_and_existing_ref() -> None:
    backend = InMemoryCAS("memory")
    configure(backend=backend)

    class ResearchState(AgentRefState):
        step: Inline[str]
        docs: Externalized[list[str]]

    state = ResearchState(step="retrieve", docs=["doc-a"])

    validate_agent_ref(state, require_externalized_exists=True)


def test_validate_checkpoint_state_rejects_unknown_fields() -> None:
    class ResearchState(AgentRefState):
        step: Inline[str]

    with pytest.raises(AgentRefError, match="unknown field"):
        validate_checkpoint_state(ResearchState, {"step": "ok", "extra": True})


def test_validate_checkpoint_state_rejects_externalized_payloads() -> None:
    class ResearchState(AgentRefState):
        step: Inline[str]
        docs: Externalized[list[str]]

    with pytest.raises(AgentRefError, match="Externalized field"):
        validate_checkpoint_state(
            ResearchState,
            {"step": "retrieve", "docs": ["raw-doc"]},
        )
