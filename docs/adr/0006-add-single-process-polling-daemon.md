# ADR-0006: Add Single Process Polling Daemon

## Status

Accepted

## Context

The project originally exposed `run-once` as a one-shot CLI that processes one Redmine issue assigned to the AI user. Operationally, the agent also needs to keep watching Redmine and process newly assigned tickets without an external scheduler repeatedly invoking the CLI.

Future work may split ticket detection and worker execution into separate processes, but that adds queueing, locking, and worker lifecycle concerns that are not needed for the current single-agent deployment.

## Decision

Add `taskboard-agent run-daemon` as a single-process polling loop while preserving `taskboard-agent run-once`.

The daemon repeatedly calls the existing one-ticket workflow. If a ticket is processed, it immediately polls again. If no assigned ticket is found, it waits for the configured interval, defaulting to 60 seconds. The daemon always searches for assigned tickets and does not accept an explicit issue ID.

Dry-run daemon execution requires a maximum iteration count so that unchanged Redmine state does not cause unbounded reprocessing of the same ticket.

## Consequences

The one-ticket workflow remains the only place that applies Redmine comments, status updates, assignee returns, and LangGraph execution. The daemon layer stays small and only controls polling, waiting, and stop behavior.

This design is simple to operate for a single daemon process, but it does not provide cross-process locking or distributed worker coordination. Those concerns must be added later if multiple daemons or separate worker processes are introduced.

## Alternatives Considered

- Replace `run-once` with a daemon-only CLI.
  - Not selected because one-shot execution remains useful for manual operation, debugging, and controlled dry-runs.
- Add a job queue and separate worker process now.
  - Not selected because the current requirement is single-process execution, and queue semantics would add unnecessary operational complexity.
- Use an external scheduler to invoke `run-once` periodically.
  - Not selected because it cannot immediately continue through a backlog without waiting for the next scheduler tick unless the scheduler becomes more complex.
