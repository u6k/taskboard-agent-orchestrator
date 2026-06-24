# ADR-0003: Keep README as the First-Read Entrypoint

## Status

Accepted

## Context

The original README contained project purpose, background, current behavior, detailed architecture, implementation plans, risks, and long-form design notes in one file.

That made the README useful as a full design dump but less useful as the first document a new reader should open. The project also needs stable places for architecture, roadmap, use cases, and agent-specific development guidance.

## Decision

Keep `README.md` focused on first-read information:

- project purpose
- current capabilities
- basic repository structure
- setup
- execution commands
- links to deeper documents

Move architecture, design, roadmap, use cases, and agent development guidance into `docs/` and `AGENTS.md`.

## Consequences

New readers can understand the project quickly from README. Detailed design has clearer ownership and can grow without making README hard to scan.

Maintainers must keep documentation links current. Detailed design should not be added back into README when a more specific document exists.

## Alternatives Considered

- Keep all design information in README.
  - Not selected because it makes README too long and mixes user-facing and implementer-facing information.
- Put all documentation under `docs/` and leave README minimal.
  - Not selected because README should still provide enough context to set up and run the project.
