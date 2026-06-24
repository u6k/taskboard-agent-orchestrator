# ADR-0002: Use Redmine REST API from Python Instead of Redmine MCP

## Status

Accepted

## Context

Redmine is the initial taskboard for this project. The agent must search assigned issues, fetch issue details and journals, add comments, update statuses, and reassign tickets.

There was a possible direction to use a Redmine MCP integration. The current implementation already has Python clients and `tool_scripts` that operate through Redmine APIs, and the desired control model keeps taskboard operations inside the orchestrator boundary.

## Decision

Operate Redmine through Python code that calls the Redmine REST API.

Do not introduce Redmine MCP as the taskboard integration mechanism. Redmine-specific operations remain in `RedmineClient`, workflow code, and explicit Python tools.

## Consequences

The orchestrator can enforce dry-run behavior, write policies, status transitions, assignee rules, and audit behavior before making Redmine changes.

The project owns more integration code and must maintain Redmine API request and response handling. If another taskboard is added later, the right direction is a taskboard adapter abstraction, not introducing Redmine MCP as a special control path.

## Alternatives Considered

- Use Redmine MCP.
  - Not selected because the project wants Redmine updates to remain under explicit Python workflow and policy control.
- Keep Redmine operations only inside generic LLM-selected tools.
  - Not selected for core workflow updates because status and assignee transitions are business rules and should not depend on unconstrained tool selection.
