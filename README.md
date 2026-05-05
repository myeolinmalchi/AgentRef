# AgentRef

Agent checkpoints should not have to carry your entire data plane.

AgentRef externalizes large workflow state into content-addressed storage
while keeping only compact references in LangGraph, LlamaIndex, AutoGen, and
Deep Agents-style checkpoints. It gives state fields two explicit roles:

- `Inline[T]`: small control state kept directly in checkpoints
- `Externalized[T]`: large values stored externally; checkpoints keep only a
  `ContentRef`

The goal is to make checkpoint write amplification hard to represent in user
code. Assigning an `Externalized` field stores the serialized payload in CAS and
keeps only a content-addressed reference in state.

## Memory Impact

Local RSS measurements show lower peak memory when large state values are stored
outside framework checkpoints. The Deep Agents row is from a real
Deep Agents-based complex workflow after obvious large-payload trimming was
already in place; the other rows are deterministic complex workflow benchmarks
that preserve final output hashes.

| Workload | Scenario | Baseline peak RSS | AgentRef peak RSS | Peak RSS reduction |
| --- | --- | ---: | ---: | ---: |
| LangGraph | Quality-preserving complex benchmark, 3-run median | 703.6 MiB | 191.6 MiB | 72.8% |
| LlamaIndex | Quality-preserving complex benchmark, 3-run median | 902.5 MiB | 229.8 MiB | 74.5% |
| AutoGen | Quality-preserving complex benchmark, 3-run median | 659.1 MiB | 243.3 MiB | 63.1% |
| Deep Agents-based complex workflow | Real workflow, 10 concurrent runs | 1.056 GiB | 803.4 MiB | 25.7% |

For the Deep Agents-based workflow, idle RSS for the application process was
effectively unchanged (267.6 MiB baseline vs. 266.5 MiB with AgentRef). On an
idle-adjusted basis, application RSS growth fell 34.0% (813.7 MiB -> 536.9 MiB).
The companion Postgres process peak fell 35.6% (331.8 MiB -> 213.6 MiB), and its
idle-adjusted RSS growth fell 41.0%. Both variants completed 10/10 runs.

These numbers are local benchmark results. They are intended to show memory
behavior for comparable before/after workloads, not to claim universal absolute
RSS values across machines.

## Install

The PyPI distribution is `agent-checkpoint-cas`; the Python import package stays
`agentref`.

With pip:

```bash
pip install agent-checkpoint-cas
```

Optional framework integrations are split by extra:

```bash
pip install "agent-checkpoint-cas[langgraph]"
pip install "agent-checkpoint-cas[llamaindex]"
pip install "agent-checkpoint-cas[autogen]"
pip install "agent-checkpoint-cas[postgres]"
pip install "agent-checkpoint-cas[all]"
```

With uv in a project:

```bash
uv add agent-checkpoint-cas
uv add "agent-checkpoint-cas[langgraph]"
uv add "agent-checkpoint-cas[llamaindex]"
uv add "agent-checkpoint-cas[autogen]"
uv add "agent-checkpoint-cas[postgres]"
uv add "agent-checkpoint-cas[all]"
```

With uv in the current environment:

```bash
uv pip install agent-checkpoint-cas
```

## Usage

### Declare State

```python
from agentref import AgentRefState, Externalized, Inline


class ResearchState(AgentRefState):
    current_step: Inline[str]
    iteration: Inline[int]
    citations: Inline[list[str]]
    retrieved_docs: Externalized[list[dict]]
    raw_html: Externalized[str]
```

### LangGraph

```python
from langgraph.graph import StateGraph

from agentref import AgentRefState, Externalized, Inline
from agentref.adapters.langgraph import LangGraphAdapter
from agentref.storage import FilesystemCAS


class RAGState(AgentRefState):
    question: Inline[str]
    docs: Externalized[list[dict]]
    answer: Inline[str]


adapter = LangGraphAdapter(
    RAGState,
    backend=FilesystemCAS("./state_blobs"),
    inline_threshold_bytes=64 * 1024,
)


def retrieve(state):
    docs = [{"id": "doc-1", "text": "large retrieved text"}]
    return {"docs": docs}


def answer(state):
    return {"answer": f"Read {len(state['docs'])} document(s)."}


graph = StateGraph(adapter.schema())
graph.add_node("retrieve", adapter.wrap_node(retrieve))
graph.add_node("answer", adapter.wrap_node(answer))
```

### LlamaIndex Workflow

```python
from agentref import AgentRefState, Externalized, Inline
from agentref.adapters.llamaindex import LlamaIndexAdapter
from agentref.storage import FilesystemCAS


class WorkflowState(AgentRefState):
    current_step: Inline[str]
    docs: Externalized[list[dict]]


adapter = LlamaIndexAdapter(
    WorkflowState,
    backend=FilesystemCAS("./state_blobs"),
)


async def retrieve_step(ctx):
    store = adapter.context_store(ctx.store)
    await store.set("current_step", "retrieve")
    await store.set("docs", [{"id": "doc-1", "text": "large retrieved text"}])
```

`await store.get("docs")` hydrates on read, while
`await store.to_checkpoint_dict()` keeps only `ContentRef` values for
externalized fields.

### AutoGen

AutoGen does not expose one stable state schema across versions. AgentRef
therefore provides explicit helpers for state dictionaries and message-history
payloads instead of monkeypatching Agent classes.

```python
from agentref.adapters.autogen import AutoGenAdapter
from agentref.storage import FilesystemCAS

adapter = AutoGenAdapter(backend=FilesystemCAS("./state_blobs"))

history = adapter.externalize_message_history(
    [{"role": "worker", "tool_result": "large tool output"}],
    threshold_bytes=1024,
)

hydrated = adapter.hydrate_message_history(history)
```

See `docs/autogen_limitations.md` for the integration boundary.

### Storage Backends

Use `FilesystemCAS` for local runs, benchmarks, and run-scoped temporary
storage. Use `PostgresCAS` when checkpoints need persistent storage, TTL
metadata, or operational cleanup.

| Backend | Best for | Lifetime model | Cleanup | Notes |
| --- | --- | --- | --- | --- |
| `InMemoryCAS` | tests and ephemeral demos | process lifetime | process exit | fastest option, not durable |
| `FilesystemCAS` | local runs, benchmarks, run-scoped workflows | directory lifetime | delete the run directory or migrate/prune by hash | simple durable storage, no built-in TTL |
| `PostgresCAS` | production, persistent checkpoints, multi-worker apps | database-managed lifetime | `expires_at` plus `prune_expired()` | TTL metadata, operational visibility, migration aliases |

All backends are passed directly to adapters:

```python
from agentref.adapters.langgraph import LangGraphAdapter
from agentref.storage import FilesystemCAS, InMemoryCAS, PostgresCAS

memory_adapter = LangGraphAdapter(RAGState, backend=InMemoryCAS())
local_adapter = LangGraphAdapter(RAGState, backend=FilesystemCAS("./state_blobs"))
postgres_adapter = LangGraphAdapter(
    RAGState,
    backend=PostgresCAS(
        dsn="postgresql://user:pass@localhost:5432/app",
        backend_id="postgres:agentref",
        default_ttl_seconds=7 * 24 * 3600,
    ),
)
```

`PostgresCAS` stores `created_at`, `last_accessed_at`, and `expires_at`
metadata. Expired payloads are removed explicitly with `backend.prune_expired()`;
AgentRef does not delete referenced objects automatically.

Existing filesystem payloads can be copied into Postgres without rewriting old
checkpoints by configuring the Postgres backend with the old backend id as an
alias:

```python
from agentref.storage import FilesystemCAS, PostgresCAS, migrate_cas

old = FilesystemCAS("./state_blobs")
new = PostgresCAS(
    dsn="postgresql://user:pass@localhost:5432/app",
    backend_id="postgres:agentref",
    backend_aliases=[old.backend_id],
)

migrate_cas(old, new)
```

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
