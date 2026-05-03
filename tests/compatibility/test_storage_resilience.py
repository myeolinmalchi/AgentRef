"""Compatibility stress checks for AgentState storage behavior."""

from __future__ import annotations

import os
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest

from agentstate import AgentState, Externalized, configure
from agentstate.adapters.autogen import AutoGenAdapter
from agentstate.core.reference import ContentRef
from agentstate.exceptions import UnresolvedReferenceError
from agentstate.storage import InMemoryCAS


class StorageCompatState(AgentState):
    """State used for backend resilience and edge-case tests."""

    payload: Externalized[bytes]


class TemporarilyUnavailableCAS(InMemoryCAS):
    """In-memory backend that simulates a transient read outage."""

    def get(self, hash: str) -> bytes:
        """Raise a transient backend failure."""

        raise TimeoutError("temporary backend outage")


@pytest.mark.compatibility
def test_backend_read_outage_is_reported_as_unresolved_reference() -> None:
    """Storage backend failures should surface through the public error type."""

    backend = InMemoryCAS(backend_id="shared")
    configure(backend=backend)
    state = StorageCompatState(payload=b"resilience-payload")
    ref = state.to_checkpoint_dict()["payload"]
    assert isinstance(ref, ContentRef)

    configure(backend=TemporarilyUnavailableCAS(backend_id="shared"))

    with pytest.raises(UnresolvedReferenceError, match="temporary backend outage"):
        ref.resolve()


@pytest.mark.compatibility
def test_concurrent_externalization_deduplicates_same_payload() -> None:
    """Concurrent writes to one backend should keep content-addressed semantics."""

    backend = InMemoryCAS()
    configure(backend=backend)
    adapter = AutoGenAdapter()
    payload = "concurrent payload " * 1024

    def externalize_once() -> str:
        message = adapter.externalize_message_history(
            [{"content": payload}],
            threshold_bytes=0,
        )[0]
        return str(message["content"]["agentstate_ref"]["hash"])

    with ThreadPoolExecutor(max_workers=8) as executor:
        hashes: List[str] = list(executor.map(lambda _: externalize_once(), range(64)))

    assert len(set(hashes)) == 1
    assert backend.object_count == 1


@pytest.mark.compatibility
def test_repeated_workflows_do_not_duplicate_or_leak_reference_payloads() -> None:
    """Run 1000 externalize/hydrate cycles and check storage/memory stability."""

    backend = InMemoryCAS()
    configure(backend=backend)
    payload = b"stable workflow payload" * 1024

    tracemalloc.start()
    try:
        for _ in range(1000):
            state = StorageCompatState(payload=payload)
            ref = state.to_checkpoint_dict()["payload"]
            assert isinstance(ref, ContentRef)
            assert ref.resolve() == payload
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert backend.object_count == 1
    assert current < 2 * 1024 * 1024
    assert peak < 4 * 1024 * 1024


@pytest.mark.compatibility
def test_exception_before_externalization_does_not_leave_orphan_blob() -> None:
    """If a workflow fails before writing an update, no CAS object is created."""

    backend = InMemoryCAS()
    configure(backend=backend)

    def failing_node() -> None:
        raise RuntimeError("node failed before externalization")

    with pytest.raises(RuntimeError, match="before externalization"):
        failing_node()

    assert backend.object_count == 0


@pytest.mark.compatibility
@pytest.mark.heavy
def test_very_large_payload_round_trip_when_enabled() -> None:
    """Optional 100MB storage edge case; disabled unless explicitly requested."""

    if os.environ.get("AGENTSTATE_RUN_HEAVY_COMPAT") != "1":
        pytest.skip("set AGENTSTATE_RUN_HEAVY_COMPAT=1 to run 100MB compatibility case")

    backend = InMemoryCAS()
    configure(backend=backend)
    payload = b"x" * (100 * 1024 * 1024)
    state = StorageCompatState(payload=payload)
    ref = state.to_checkpoint_dict()["payload"]

    assert isinstance(ref, ContentRef)
    assert ref.resolve() == payload
    assert backend.object_count == 1
