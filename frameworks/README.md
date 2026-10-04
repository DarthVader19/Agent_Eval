# Framework comparison mapping

All adapters target one worker rather than adding framework-specific planner/critic agents:

| Framework | Mapping | Shared tools | State/persistence | Current caveat |
|---|---|---|---|---|
| LangGraph | Explicit `StateGraph`: one action node, conditional continue/end edge | Direct shared `ToolBox` | Per-run `MemorySaver` checkpoint | Primary architecture baseline |
| CrewAI | One sequential `Agent` and one `Task`, delegation disabled | Decorated wrappers call `ToolBox.invoke` | Framework task lifecycle only; harness stores complete trace rows | Framework token accounting is estimated |
| AutoGen | One `AssistantAgent` in a one-participant `RoundRobinGroupChat`, bounded turns | Callable wrappers call `ToolBox.invoke` | Per-run team; harness stores complete trace rows | Requires `autogen-ext[openai]`; framework token accounting is estimated |

For an apples-to-apples run, the CLI gives each adapter the same model role, temperature, output-token limit, tool object, task budget, task set and repeat count. Framework-required system/prompt framing differs; it is a recorded limitation rather than a hidden claim of identical tokenization.

## Qualitative implementation comparison

Scores below are implementation-time assessments (1 = low, 5 = high), not measured benchmark outcomes.

| Framework | LOC / implementation cost | Debuggability / trace quality | Orchestration flexibility | State/persistence | Ecosystem maturity | Runaway-loop controls |
|---|---:|---:|---:|---:|---:|---:|
| LangGraph | 3 | 5 | 5 | 5 | 4 | 5 |
| CrewAI | 3 | 3 | 4 | 3 | 4 | 4 |
| AutoGen | 3 | 4 | 4 | 3 | 4 | 4 |

Framework comparisons are not asserted as complete until all optional dependencies import in the target environment and the same live pilot has been run for each adapter.
