# ADR-0005: Use LangChain Tools and Agent Loop

## Status

Accepted

## Context

The project originally defined tool schemas with local `ToolSpec` objects and executed function calls with a local `FunctionCallingAgent`. Since the agent runtime already depends on LangGraph, maintaining a separate schema and function calling loop made the implementation more custom than necessary.

The project still needs local business policy for dry-run, write tools, approval-required operations, Redmine status transitions, and taskboard ownership rules.

## Decision

Define tools as LangChain `BaseTool` instances created with `@tool`, type hints, and docstrings. Tool scripts expose `create_tool(context)` and keep business metadata such as `risk`, `planner_visible`, and `dry_run_safe` in `BaseTool.extras`.

Use LangChain's LangGraph-backed agent harness for the LLM tool calling loop. Keep local policy checks around tool execution instead of delegating Redmine workflow rules to LangChain.

## Consequences

Tool schema generation and tool call message handling are handled by LangChain. The local code keeps only catalog loading, planner visibility, write/dry-run/approval policy, and conversion of tool outputs into project artifacts.

Existing scripted skills can keep calling `context.execute_tool(...)`, but that method now invokes LangChain tools under the same local policy checks.

## Alternatives Considered

- Keep `ToolSpec`, `ToolRegistry`, and the local function calling loop.
  - Not selected because it duplicates LangChain/LangGraph behavior and makes the agent runtime harder to align with the framework.
- Move Redmine workflow updates fully into LangChain tools.
  - Not selected because taskboard status changes, assignee changes, dry-run behavior, and approval policy are business controls that should remain in the orchestrator boundary.
