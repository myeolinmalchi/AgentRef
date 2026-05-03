"""LlamaIndex Workflow-style example using AgentRef Context stores."""

from __future__ import annotations

from typing import Any, Dict

from agentref import AgentRefState, Externalized, Inline
from agentref.adapters.llamaindex import LlamaIndexAdapter
from agentref.storage import FilesystemCAS


class ResearchWorkflowState(AgentRefState):
    """State for a research workflow."""

    current_step: Inline[str]
    docs: Externalized[list[dict[str, str]]]
    summary: Inline[str]


async def retrieve_step(ctx: Any, adapter: LlamaIndexAdapter) -> None:
    """Store retrieved documents through a Context.store proxy."""

    store = adapter.context_store(ctx.store)
    store["current_step"] = "retrieve"
    store["docs"] = [{"id": "doc-1", "text": "large research document"}]


async def summarize_step(ctx: Any, adapter: LlamaIndexAdapter) -> None:
    """Hydrate retrieved documents and write a small summary."""

    store = adapter.context_store(ctx.store)
    docs = store["docs"]
    store["current_step"] = "summarize"
    store["summary"] = f"Summarized {len(docs)} document(s)."


class DictContext:
    """Tiny Context stand-in for running this example without LlamaIndex."""

    def __init__(self) -> None:
        """Create an in-memory store."""

        self.store: Dict[str, Any] = {}


async def main() -> None:
    """Run the workflow steps with a dict-backed Context stand-in."""

    adapter = LlamaIndexAdapter(
        ResearchWorkflowState,
        backend=FilesystemCAS(root="./state_blobs"),
    )
    ctx = DictContext()
    await retrieve_step(ctx, adapter)
    await summarize_step(ctx, adapter)
    checkpoint = adapter.context_store(ctx.store).to_checkpoint_dict()
    print(checkpoint["current_step"])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
