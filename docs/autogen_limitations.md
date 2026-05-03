# AutoGen Adapter Limitations

AutoGen does not expose one stable, universal state schema equivalent to
LangGraph's `StateGraph` schema or LlamaIndex Workflow's `Context.store`.

The `agentstate` AutoGen adapter therefore supports two explicit integration
surfaces:

- declared `AgentState` mappings via `externalize_state()` and `hydrate_state()`
- conversation-history dictionaries via `externalize_message_history()` and
  `hydrate_message_history()`

The adapter intentionally does not monkeypatch AutoGen Agent classes. Projects
using a concrete AutoGen version can call these helpers at their state
save/restore boundary or around large tool-result/message-history writes.
