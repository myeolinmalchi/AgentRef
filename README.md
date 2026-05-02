# AgentState

AgentState is a small Python library for keeping large data-plane values out of
LLM agent framework checkpoints. It gives state fields two explicit roles:

- `Inline[T]`: control-plane values stored directly in framework state
- `Externalized[T]`: data-plane values stored in content-addressed storage while
  the checkpoint keeps only a `ContentRef`

The goal is to make checkpoint write amplification hard to represent in user
code. Assigning an `Externalized` field stores the serialized payload in CAS and
keeps only a content-addressed reference in state.

## Memory Impact

Local RSS measurements show lower peak memory when large state values are stored
outside framework checkpoints. The Deep Agents row is from a real
Deep Agents-based complex workflow after obvious large-payload trimming was
already in place; the other rows are deterministic complex workflow benchmarks
that preserve final output hashes.

| Workload | Scenario | Baseline peak RSS | AgentState peak RSS | Peak RSS reduction |
| --- | --- | ---: | ---: | ---: |
| LangGraph | Quality-preserving complex benchmark, 3-run median | 703.6 MiB | 191.6 MiB | 72.8% |
| LlamaIndex | Quality-preserving complex benchmark, 3-run median | 902.5 MiB | 229.8 MiB | 74.5% |
| AutoGen | Quality-preserving complex benchmark, 3-run median | 659.1 MiB | 243.3 MiB | 63.1% |
| Deep Agents-based complex workflow | Real workflow, 10 concurrent runs | 1.056 GiB | 803.4 MiB | 25.7% |

For the Deep Agents-based workflow, idle RSS for the application process was
effectively unchanged (267.6 MiB baseline vs. 266.5 MiB with AgentState). On an
idle-adjusted basis, application RSS growth fell 34.0% (813.7 MiB -> 536.9 MiB).
The companion Postgres process peak fell 35.6% (331.8 MiB -> 213.6 MiB), and its
idle-adjusted RSS growth fell 41.0%. Both variants completed 10/10 runs.

These numbers are local benchmark results. They are intended to show memory
behavior for comparable before/after workloads, not to claim universal absolute
RSS values across machines.

## Install

```bash
pip install agentstate
```

Optional framework integrations are split by extra:

```bash
pip install "agentstate[langgraph]"
pip install "agentstate[llamaindex]"
pip install "agentstate[autogen]"
pip install "agentstate[all]"
```

## Usage

### Core State Model

```python
from agentstate import AgentState, Externalized, Inline, configure
from agentstate.storage import FilesystemCAS


class ResearchState(AgentState):
    current_step: Inline[str]
    iteration: Inline[int]
    citations: Inline[list[str]]
    retrieved_docs: Externalized[list[dict]]
    raw_html: Externalized[str]


configure(
    backend=FilesystemCAS(root="./state_blobs"),
    inline_threshold_bytes=64 * 1024,
)

state = ResearchState(
    current_step="retrieve",
    iteration=1,
    citations=[],
    retrieved_docs=[{"id": "doc-1", "text": "..."}],
    raw_html="<html>...</html>",
)

checkpoint = state.to_checkpoint_dict()
assert checkpoint["retrieved_docs"].__class__.__name__ == "ContentRef"
assert state.retrieved_docs == [{"id": "doc-1", "text": "..."}]
```

### LangGraph

```python
from langgraph.graph import StateGraph

from agentstate import AgentState, Externalized, Framework, Inline
from agentstate.adapters import auto_adapt
from agentstate.adapters.langgraph import LangGraphAdapter


class RAGState(AgentState):
    question: Inline[str]
    docs: Externalized[list[dict]]


StateSchema = auto_adapt(RAGState, Framework.LANGGRAPH)
adapter = LangGraphAdapter()


def retrieve(state):
    docs = [{"id": "doc-1", "text": "large retrieved text"}]
    return adapter.externalize_node_update(RAGState, {"docs": docs})


def answer(state):
    hydrated = adapter.hydrate_state_for_node(RAGState, state)
    return {"question": hydrated["question"]}


graph = StateGraph(StateSchema)
graph.add_node("retrieve", retrieve)
graph.add_node("answer", answer)
```

### LlamaIndex Workflow

```python
from agentstate import AgentState, Externalized, Inline
from agentstate.adapters.llamaindex import LlamaIndexAdapter


class WorkflowState(AgentState):
    current_step: Inline[str]
    docs: Externalized[list[dict]]


adapter = LlamaIndexAdapter()


async def retrieve_step(ctx):
    store = adapter.context_store(WorkflowState, ctx.store)
    store["current_step"] = "retrieve"
    store["docs"] = [{"id": "doc-1", "text": "large retrieved text"}]
```

`store["docs"]` hydrates on read, while `store.to_checkpoint_dict()` keeps only
`ContentRef` values for externalized fields.

### AutoGen

AutoGen does not expose one stable state schema across versions. AgentState
therefore provides explicit helpers for state dictionaries and message-history
payloads instead of monkeypatching Agent classes.

```python
from agentstate.adapters.autogen import AutoGenAdapter

adapter = AutoGenAdapter()

history = adapter.externalize_message_history(
    [{"role": "worker", "tool_result": "large tool output"}],
    threshold_bytes=1024,
)

hydrated = adapter.hydrate_message_history(history)
```

See `docs/autogen_limitations.md` for the integration boundary.

## Invariants

The test suite covers these core invariants:

- externalized payload bytes do not appear in checkpoint bytes
- identical payloads produce identical content hashes
- oversized inline values raise `InlineSizeExceeded`
- checkpoint round trips preserve hydrated values
- ambiguous framework auto-detection raises a clear error
- older checkpoints can hydrate externalized values while CAS content exists

## Examples

- `examples/langgraph_rag.py`
- `examples/llamaindex_research_workflow.py`
- `examples/autogen_multi_agent.py`

Each example avoids importing optional framework packages at module import time,
so the repository can be imported and tested without installing every framework.
