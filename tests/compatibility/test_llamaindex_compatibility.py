"""Real LlamaIndex Workflow compatibility checks for AgentState adapters."""

# mypy: disable-error-code="import-not-found,untyped-decorator,valid-type,attr-defined,no-any-return"

import asyncio
import pickle
from typing import Any, Dict, List, Optional, Tuple

import pytest

from agentstate import AgentState, Externalized, Inline, configure
from agentstate.adapters.llamaindex import LlamaIndexAdapter
from agentstate.storage import InMemoryCAS


RAW_PDF = b"%PDF-compat-llamaindex%" + (b"x" * 262_144)
RAW_NOTE = "llamaindex compatibility research note " * 4096
RAW_DOCS = [{"id": "paper-1", "text": "paper text " * 2048}]
RAW_NESTED = {"sections": [{"title": "intro", "body": RAW_NOTE}]}


class LlamaIndexCompatState(AgentState):
    """AgentState schema used with real LlamaIndex Workflow Context stores."""

    phase: Inline[str]
    iteration: Inline[int]
    pdf: Externalized[bytes]
    docs: Externalized[List[Dict[str, str]]]
    notes: Externalized[str]
    nested: Externalized[Dict[str, Any]]
    maybe: Externalized[Optional[str]]


class LlamaIndexParallelState(AgentState):
    """State used for parallel workflow step compatibility."""

    phase: Inline[str]
    branch_a: Externalized[str]
    branch_b: Externalized[str]


class LlamaIndexHumanState(AgentState):
    """State used for human-in-the-loop workflow compatibility."""

    status: Inline[str]
    draft: Externalized[str]


def _workflow_imports() -> Tuple[Any, Any, Any, Any, Any]:
    """Import LlamaIndex Workflow APIs, skipping known incompatible runtimes."""

    try:
        from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
    except Exception as exc:
        pytest.skip(f"LlamaIndex Workflow import failed in this runtime: {exc}")
    return Workflow, Context, StartEvent, StopEvent, step


@pytest.mark.compatibility
def test_llamaindex_context_round_trip_excludes_externalized_payloads() -> None:
    """Check single run, restored Context, and multi-run Context reuse."""

    Workflow, Context, StartEvent, StopEvent, step = _workflow_imports()
    adapter = LlamaIndexAdapter()
    backend = InMemoryCAS()
    configure(backend=backend)

    class ResearchWorkflow(Workflow):  # type: ignore[misc]
        @step
        async def retrieve(self, ctx: Context, ev: StartEvent) -> StopEvent:
            store = adapter.context_store(LlamaIndexCompatState, ctx.store)
            previous_iteration = await store.get("iteration", 0)
            next_iteration = previous_iteration + 1

            await store.set("phase", "retrieved")
            await store.set("iteration", next_iteration)
            await store.set("pdf", RAW_PDF)
            await store.set("docs", RAW_DOCS)
            await store.set("notes", RAW_NOTE)
            await store.set("nested", RAW_NESTED)
            await store.set("maybe", None)

            assert await store.get("pdf") == RAW_PDF
            assert await store.get("docs") == RAW_DOCS
            assert await store.get("notes") == RAW_NOTE
            assert await store.get("nested") == RAW_NESTED
            assert await store.get("maybe") is None
            return StopEvent(result={"iteration": next_iteration})

    async def run_scenario() -> None:
        workflow = ResearchWorkflow(timeout=10)

        first_handler = workflow.run()
        assert await first_handler == {"iteration": 1}

        context_dict = first_handler.ctx.to_dict()
        serialized_context = pickle.dumps(context_dict)
        assert RAW_PDF not in serialized_context
        assert RAW_NOTE.encode() not in serialized_context

        restored_context = Context.from_dict(workflow, context_dict)
        restored_store = adapter.context_store(
            LlamaIndexCompatState,
            restored_context.store,
        )
        assert await restored_store.get("pdf") == RAW_PDF
        assert await restored_store.get("docs") == RAW_DOCS
        assert await restored_store.get("nested") == RAW_NESTED

        second_handler = workflow.run(ctx=restored_context)
        assert await second_handler == {"iteration": 2}

    asyncio.run(run_scenario())

    assert backend.object_count == 5


@pytest.mark.compatibility
def test_llamaindex_parallel_steps_share_externalized_context_state() -> None:
    """Cover parallel step fan-out and join with hydrated Context state."""

    Workflow, Context, StartEvent, StopEvent, step = _workflow_imports()
    try:
        from llama_index.core.workflow import Event
    except Exception as exc:
        pytest.skip(f"LlamaIndex Event import failed: {exc}")

    adapter = LlamaIndexAdapter()
    backend = InMemoryCAS()
    configure(backend=backend)

    class BranchEvent(Event):  # type: ignore[misc]
        label: str

    class DoneEvent(Event):  # type: ignore[misc]
        label: str

    class ParallelWorkflow(Workflow):  # type: ignore[misc]
        @step
        async def start(self, ctx: Context, ev: StartEvent) -> BranchEvent:
            store = adapter.context_store(LlamaIndexParallelState, ctx.store)
            await store.set("phase", "started")
            ctx.send_event(BranchEvent(label="a"))
            return BranchEvent(label="b")

        @step
        async def worker(self, ctx: Context, ev: BranchEvent) -> DoneEvent:
            store = adapter.context_store(LlamaIndexParallelState, ctx.store)
            field = "branch_a" if ev.label == "a" else "branch_b"
            await store.set(field, f"parallel-payload-{ev.label}")
            return DoneEvent(label=ev.label)

        @step
        async def join(self, ctx: Context, ev: DoneEvent) -> Optional[StopEvent]:
            events = ctx.collect_events(ev, [DoneEvent, DoneEvent])
            if events is None:
                return None
            store = adapter.context_store(LlamaIndexParallelState, ctx.store)
            return StopEvent(
                result=[
                    await store.get("branch_a"),
                    await store.get("branch_b"),
                ]
            )

    async def run_scenario() -> None:
        handler = ParallelWorkflow(timeout=10).run()
        assert await handler == ["parallel-payload-a", "parallel-payload-b"]
        serialized_context = pickle.dumps(handler.ctx.to_dict())
        assert b"parallel-payload-a" not in serialized_context
        assert b"parallel-payload-b" not in serialized_context

    asyncio.run(run_scenario())

    assert backend.object_count == 2


@pytest.mark.compatibility
def test_llamaindex_human_in_the_loop_keeps_draft_externalized() -> None:
    """Cover InputRequiredEvent/HumanResponseEvent with Context state hydration."""

    Workflow, Context, StartEvent, StopEvent, step = _workflow_imports()
    try:
        from llama_index.core.workflow import HumanResponseEvent, InputRequiredEvent
    except Exception as exc:
        pytest.skip(f"LlamaIndex human-in-the-loop imports failed: {exc}")

    adapter = LlamaIndexAdapter()
    backend = InMemoryCAS()
    configure(backend=backend)
    draft = "human review draft " * 4096

    class ReviewRequired(InputRequiredEvent):  # type: ignore[misc]
        prompt: str

    class ReviewResponse(HumanResponseEvent):  # type: ignore[misc]
        response: str

    class ReviewWorkflow(Workflow):  # type: ignore[misc]
        @step
        async def ask(self, ctx: Context, ev: StartEvent) -> ReviewRequired:
            store = adapter.context_store(LlamaIndexHumanState, ctx.store)
            await store.set("status", "waiting")
            await store.set("draft", draft)
            return ReviewRequired(prompt="approve")

        @step
        async def answer(self, ctx: Context, ev: ReviewResponse) -> StopEvent:
            store = adapter.context_store(LlamaIndexHumanState, ctx.store)
            assert await store.get("draft") == draft
            await store.set("status", ev.response)
            return StopEvent(result=ev.response)

    async def run_scenario() -> None:
        handler = ReviewWorkflow(timeout=10).run()
        async for event in handler.stream_events():
            if isinstance(event, ReviewRequired):
                assert draft.encode() not in pickle.dumps(handler.ctx.to_dict())
                handler.ctx.send_event(ReviewResponse(response="approved"))
        assert await handler == "approved"

    asyncio.run(run_scenario())

    assert backend.object_count == 1
