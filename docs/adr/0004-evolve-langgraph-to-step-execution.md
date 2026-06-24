# ADR-0004: Evolve LangGraph to Step-Level Execution State

## Status

Accepted

## Context

The current implementation uses LangGraph for ticket-level conversation state, checkpointing, feedback ingestion, and revision flow. However, the actual task steps are still executed inside `TaskOrchestrator._execute_steps()` as a Python for loop.

This means LangGraph sees the larger execution node, but not each individual step as a checkpointed state transition. For long-running or multi-step work, the project needs clearer visibility into which step is pending, running, completed, failed, skipped, or waiting for a human.

## Decision

Evolve LangGraph from a coarse ticket conversation wrapper into the step-level execution state machine.

The target graph should store planned steps and route execution through nodes such as:

- `plan`
- `publish_plan`
- `select_next_step`
- `execute_step`
- `record_step_result`
- `route_after_step`
- `finalize_success`
- `finalize_failed`
- `wait_for_human`

`TaskOrchestrator` should keep plan creation and one-step execution helpers, but LangGraph should eventually own the step loop and checkpoint step results.

## Consequences

The system will be able to checkpoint after individual steps, inspect progress from graph state, resume from unfinished work, and reason about human feedback against specific completed or failed steps.

The migration must be phased. First add step state, then extract one-step execution, then move the loop into LangGraph. A single large rewrite would make existing Redmine update behavior and revision handling too easy to break.

## Alternatives Considered

- Keep step execution entirely inside `TaskOrchestrator._execute_steps()`.
  - Not selected because it hides progress from LangGraph and limits restartability.
- Move all tool execution directly to LangGraph `ToolNode` immediately.
  - Not selected because the project first needs stable step state and business policy boundaries. ToolNode migration can be evaluated later, starting with low-risk read tools.
