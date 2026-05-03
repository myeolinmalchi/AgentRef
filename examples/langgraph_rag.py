"""LangGraph RAG example using AgentState externalized fields."""

from __future__ import annotations

from typing import Any, Dict, List

from agentstate import AgentState, Externalized, Inline
from agentstate.adapters.langgraph import LangGraphAdapter
from agentstate.storage import FilesystemCAS


class RAGState(AgentState):
    """State for a simple retrieval-augmented generation graph."""

    question: Inline[str]
    docs: Externalized[List[Dict[str, str]]]
    answer: Inline[str]


def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return retrieved documents."""

    docs = [
        {
            "id": "doc-1",
            "text": "AgentState stores large documents outside checkpoints.",
        }
    ]
    return {"docs": docs}


def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a small inline answer from hydrated documents."""

    docs = state["docs"]
    return {"answer": f"Used {len(docs)} retrieved document(s)."}


def build_graph(adapter: LangGraphAdapter) -> Any:
    """Build and return a LangGraph StateGraph.

    This function imports LangGraph lazily so the example module can be imported
    without optional dependencies installed.
    """

    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(adapter.schema())
    graph.add_node("retrieve", adapter.wrap_node(retrieve_node))
    graph.add_node("answer", adapter.wrap_node(answer_node))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph


def main() -> None:
    """Run the graph when LangGraph is installed."""

    adapter = LangGraphAdapter(RAGState, backend=FilesystemCAS(root="./state_blobs"))
    graph = build_graph(adapter).compile()
    result = graph.invoke({"question": "How does AgentState avoid bloat?"})
    print(result["answer"])


if __name__ == "__main__":
    main()
