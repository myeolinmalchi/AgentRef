"""AutoGen multi-agent style example using message-history externalization."""

from __future__ import annotations

from typing import Any, Dict, List

from agentref.adapters.autogen import AutoGenAdapter
from agentref.storage import FilesystemCAS


def planner_message(task: str) -> Dict[str, str]:
    """Return a small planner message."""

    return {"role": "planner", "content": f"Plan work for: {task}"}


def worker_tool_message() -> Dict[str, str]:
    """Return a large tool-result message."""

    return {
        "role": "worker",
        "tool_result": "large tool result " * 256,
    }


def run_conversation(adapter: AutoGenAdapter) -> List[Dict[str, Any]]:
    """Run a small multi-agent conversation history through AgentRef."""

    messages = [planner_message("research"), worker_tool_message()]
    checkpoint_messages = adapter.externalize_message_history(
        messages,
        threshold_bytes=1024,
    )
    return adapter.hydrate_message_history(checkpoint_messages)


def main() -> None:
    """Run the example conversation."""

    adapter = AutoGenAdapter(backend=FilesystemCAS(root="./state_blobs"))
    hydrated = run_conversation(adapter)
    print(hydrated[-1]["tool_result"][:32])


if __name__ == "__main__":
    main()
