"""Real LangGraph compatibility checks for AgentRefState adapters."""

import asyncio
import os
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, List, Optional, TypedDict, cast

import pytest

from agentref import AgentRefState, Externalized, Inline, configure
from agentref.adapters.langgraph import LangGraphAdapter
from agentref.core.reducers import ref_aware_list_append
from agentref.storage import InMemoryCAS


RAW_BYTES = b"langgraph-compat-raw-bytes-" * 4096
RAW_TEXT = "langgraph compat retrieved document " * 2048
RAW_NESTED = {"outer": {"inner": ["nested", RAW_TEXT]}}


class LangGraphCompatState(AgentRefState):
    """AgentRefState schema used by real LangGraph compatibility tests."""

    phase: Inline[str]
    count: Inline[int]
    payload: Externalized[bytes]
    docs: Externalized[List[Dict[str, str]]]
    nested: Externalized[Dict[str, Any]]


class SinglePayloadState(AgentRefState):
    """Small helper state for graph pattern tests."""

    payload: Externalized[str]


class BranchCycleState(AgentRefState):
    """State used for branch, cycle, and async node compatibility."""

    route: Inline[str]
    count: Inline[int]
    payload: Externalized[str]


@pytest.mark.compatibility
@pytest.mark.parametrize("checkpointer_name", ["memory", "sqlite"])
def test_langgraph_checkpointers_preserve_externalized_payloads(
    checkpointer_name: str,
    tmp_path: Path,
) -> None:
    """Check real LangGraph checkpointing, hydration, and time-travel snapshots."""

    graph_mod = pytest.importorskip("langgraph.graph")
    memory_mod = pytest.importorskip("langgraph.checkpoint.memory")
    adapter = LangGraphAdapter()
    backend = InMemoryCAS()
    configure(backend=backend)

    schema = adapter.wrap_state_class(LangGraphCompatState)

    def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return adapter.externalize_node_update(
            LangGraphCompatState,
            {
                "phase": "retrieved",
                "count": state["count"] + 1,
                "payload": RAW_BYTES,
                "docs": [{"id": "doc-1", "text": RAW_TEXT}],
                "nested": RAW_NESTED,
            },
        )

    def analyze_node(state: Dict[str, Any]) -> Dict[str, Any]:
        hydrated = adapter.hydrate_state_for_node(LangGraphCompatState, state)
        assert hydrated["payload"] == RAW_BYTES
        assert hydrated["docs"][0]["text"] == RAW_TEXT
        assert hydrated["nested"] == RAW_NESTED
        return {"phase": "done", "count": hydrated["count"] + 1}

    graph = graph_mod.StateGraph(schema)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_edge(graph_mod.START, "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", graph_mod.END)

    db_path = tmp_path / "langgraph-checkpoints.sqlite"
    if checkpointer_name == "memory":
        checkpointer_cm = nullcontext(memory_mod.InMemorySaver())
    else:
        sqlite_mod = pytest.importorskip("langgraph.checkpoint.sqlite")
        checkpointer_cm = sqlite_mod.SqliteSaver.from_conn_string(str(db_path))

    with checkpointer_cm as checkpointer:
        if hasattr(checkpointer, "setup"):
            checkpointer.setup()
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": f"compat-{checkpointer_name}"}}

        result = app.invoke({"phase": "start", "count": 0}, config)
        current_snapshot = app.get_state(config)
        history = list(app.get_state_history(config))

        serialized_snapshot = pickle.dumps(current_snapshot.values)
        assert RAW_BYTES not in serialized_snapshot
        assert RAW_TEXT.encode() not in serialized_snapshot

        hydrated_result = adapter.hydrate_state_for_node(
            LangGraphCompatState,
            result,
        )
        assert hydrated_result["payload"] == RAW_BYTES
        assert hydrated_result["docs"][0]["text"] == RAW_TEXT
        assert hydrated_result["nested"] == RAW_NESTED
        assert backend.object_count == 3

        retrieved_snapshot = next(
            snapshot for snapshot in history if snapshot.values.get("phase") == "retrieved"
        )
        hydrated_old = adapter.hydrate_state_for_node(
            LangGraphCompatState,
            retrieved_snapshot.values,
        )
        assert hydrated_old["phase"] == "retrieved"
        assert hydrated_old["payload"] == RAW_BYTES

    if checkpointer_name == "sqlite":
        assert RAW_BYTES not in db_path.read_bytes()
        assert RAW_TEXT.encode() not in db_path.read_bytes()


@pytest.mark.compatibility
def test_langgraph_branch_cycle_and_async_node_patterns() -> None:
    """Cover conditional branches, a cycle, and async node execution."""

    graph_mod = pytest.importorskip("langgraph.graph")
    adapter = LangGraphAdapter()
    backend = InMemoryCAS()
    configure(backend=backend)
    schema = adapter.wrap_state_class(BranchCycleState)

    def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"route": "left", "count": state["count"]}

    def choose_branch(state: Dict[str, Any]) -> str:
        return str(state["route"])

    async def left_node(state: Dict[str, Any]) -> Dict[str, Any]:
        hydrated = adapter.hydrate_state_for_node(BranchCycleState, state)
        next_count = hydrated["count"] + 1
        return adapter.externalize_node_update(
            BranchCycleState,
            {
                "route": "left",
                "count": next_count,
                "payload": f"loop-payload-{next_count}",
            },
        )

    def right_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return adapter.externalize_node_update(
            BranchCycleState,
            {"route": "right", "count": 1, "payload": "right-payload"},
        )

    def loop_or_end(state: Dict[str, Any]) -> str:
        return "left" if state["count"] < 2 else graph_mod.END

    graph = graph_mod.StateGraph(schema)
    graph.add_node("start", start_node)
    graph.add_node("left", left_node)
    graph.add_node("right", right_node)
    graph.add_edge(graph_mod.START, "start")
    graph.add_conditional_edges("start", choose_branch, {"left": "left", "right": "right"})
    graph.add_conditional_edges("left", loop_or_end, {"left": "left", graph_mod.END: graph_mod.END})
    graph.add_edge("right", graph_mod.END)

    result = asyncio.run(graph.compile().ainvoke({"route": "left", "count": 0}))
    hydrated = adapter.hydrate_state_for_node(BranchCycleState, result)

    assert hydrated == {"route": "left", "count": 2, "payload": "loop-payload-2"}
    assert backend.object_count == 2


@pytest.mark.compatibility
def test_langgraph_send_fanout_deduplicates_ref_wrappers() -> None:
    """Exercise Send API dynamic fan-out and ref-aware list reduction."""

    graph_mod = pytest.importorskip("langgraph.graph")
    types_mod = pytest.importorskip("langgraph.types")
    adapter = LangGraphAdapter()
    backend = InMemoryCAS()
    configure(backend=backend)

    class FanoutState(TypedDict, total=False):
        items: List[str]
        payload_refs: Annotated[List[Dict[str, Any]], ref_aware_list_append]

    def route_to_workers(state: FanoutState) -> List[Any]:
        return [
            types_mod.Send("worker", {"item": item})
            for item in [*state["items"], state["items"][0]]
        ]

    def worker(state: Dict[str, str]) -> Dict[str, Any]:
        ref_wrapper = adapter.externalize_node_update(
            SinglePayloadState,
            {"payload": f"payload-for-{state['item']}"},
        )["payload"]
        return {"payload_refs": [ref_wrapper]}

    graph = graph_mod.StateGraph(FanoutState)
    graph.add_node("worker", worker)
    graph.add_conditional_edges(graph_mod.START, route_to_workers, ["worker"])
    graph.add_edge("worker", graph_mod.END)

    result = graph.compile().invoke({"items": ["a", "b"]})

    assert len(result["payload_refs"]) == 2
    assert backend.object_count == 2


@pytest.mark.compatibility
def test_langgraph_dataclass_pydantic_and_subgraph_patterns_accept_wrappers() -> None:
    """Cover non-TypedDict state patterns plus subgraph invocation."""

    graph_mod = pytest.importorskip("langgraph.graph")
    pydantic = pytest.importorskip("pydantic")
    adapter = LangGraphAdapter()
    backend = InMemoryCAS()
    configure(backend=backend)

    @dataclass
    class DataclassState:
        phase: str
        payload: Optional[Dict[str, Any]] = None

    BaseModel = pydantic.BaseModel

    class PydanticState(BaseModel):  # type: ignore[misc, valid-type]
        phase: str
        payload: Optional[Dict[str, Any]] = None

    for state_schema in (DataclassState, PydanticState):

        def node(state: Any) -> Dict[str, Any]:
            return {
                "phase": "done",
                "payload": adapter.externalize_node_update(
                    SinglePayloadState,
                    {"payload": "native-schema-payload"},
                )["payload"],
            }

        graph = graph_mod.StateGraph(state_schema)
        graph.add_node("node", node)
        graph.add_edge(graph_mod.START, "node")
        graph.add_edge("node", graph_mod.END)
        result = graph.compile().invoke({"phase": "start"})

        hydrated = adapter.hydrate_mapping(
            SinglePayloadState,
            {"payload": result["payload"]},
        )
        assert hydrated["payload"] == "native-schema-payload"

    schema = adapter.wrap_state_class(SinglePayloadState)

    def child_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return adapter.externalize_node_update(
            SinglePayloadState,
            {"payload": "subgraph-payload"},
        )

    child = graph_mod.StateGraph(schema)
    child.add_node("child", child_node)
    child.add_edge(graph_mod.START, "child")
    child.add_edge("child", graph_mod.END)
    child_app = child.compile()

    def parent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return cast(Dict[str, Any], child_app.invoke(state))

    parent = graph_mod.StateGraph(schema)
    parent.add_node("parent", parent_node)
    parent.add_edge(graph_mod.START, "parent")
    parent.add_edge("parent", graph_mod.END)

    result = parent.compile().invoke({})

    assert adapter.hydrate_state_for_node(SinglePayloadState, result)["payload"] == (
        "subgraph-payload"
    )


@pytest.mark.compatibility
@pytest.mark.heavy
def test_langgraph_hydrates_very_large_payload_when_enabled() -> None:
    """Optional 100MB edge-case check; disabled unless explicitly requested."""

    if os.environ.get("AGENTREF_RUN_HEAVY_COMPAT") != "1":
        pytest.skip("set AGENTREF_RUN_HEAVY_COMPAT=1 to run 100MB compatibility case")

    graph_mod = pytest.importorskip("langgraph.graph")
    adapter = LangGraphAdapter()
    configure(backend=InMemoryCAS())
    payload = b"x" * (100 * 1024 * 1024)
    schema = adapter.wrap_state_class(SinglePayloadState)

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        return adapter.externalize_node_update(SinglePayloadState, {"payload": payload})

    graph = graph_mod.StateGraph(schema)
    graph.add_node("node", node)
    graph.add_edge(graph_mod.START, "node")
    graph.add_edge("node", graph_mod.END)

    result = graph.compile().invoke({})

    assert adapter.hydrate_state_for_node(SinglePayloadState, result)["payload"] == payload
