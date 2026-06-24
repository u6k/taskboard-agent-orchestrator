# ADR-0001: Use a Self-Managed Orchestrator

## Status

Accepted

## Context

This project needs to coordinate taskboard state, human review, tool permissions, dry-run behavior, skill execution, progress comments, and durable execution history.

An external execution platform such as OpenClaw could provide an execution runtime, but the core problem in this project is not only running an agent. The project must preserve business-level control over how tickets are selected, how work is planned, when Redmine is updated, when a human must decide, and how execution resumes after feedback.

## Decision

Use a self-managed orchestrator as the primary control layer.

LangGraph is used inside this orchestrator for durable ticket conversations and, over time, step-level execution state. OpenClaw is not adopted as the execution foundation.

## Consequences

The project keeps direct control over workflow rules, Redmine updates, dry-run handling, permission checks, checkpoint structure, and human-in-the-loop behavior.

This increases implementation responsibility inside this repository. Scheduling, recovery, step execution, and tool policy must be designed and tested here instead of delegated to an external runtime.

## Alternatives Considered

- Use OpenClaw as the execution foundation.
  - Not selected because it would move too much workflow control outside this repository while the project still needs explicit taskboard, review, and permission semantics.
- Use only one-shot LLM calls without an orchestrator.
  - Not selected because the target workflow requires durable state, feedback handling, and restartable multi-step execution.
