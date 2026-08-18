# Architecture Decision Records

Architecture Decision Record、ADRは、アーキテクチャ上の重要な意思決定を短く記録する文書である。

## 目的

ADRの目的は、後から読んだ人が「なぜその方針にしたのか」を理解できるようにすることである。

コードや設計書は現在の構造を説明するが、判断時の背景、検討した選択肢、採用しなかった理由は失われやすい。ADRでは、実装者が将来迷いそうな判断について、背景、決定、影響、代替案を残す。

## 追記するタイミング

ADRは、将来の設計や実装に影響する判断をしたときに追加する。

追加すべき例:

- 実行基盤、永続化方式、タスクボード連携方式の選択
- LangGraph、LLM、tool実行、スキル実行など中核構造の方針
- README、docs、AGENTS.mdなど文書体系の方針
- セキュリティ、権限、dry-run、人間承認に関わる方針
- 一見自然な別案を採用しないと決めた判断

追加しなくてよい例:

- 関数名や変数名の細かな選択
- 小さなバグ修正
- 一時的なTODO
- 実装メモだけで済む内容
- PR内で完結する局所的な変更

判断を変更する場合は、既存ADRを書き換えて履歴を消すのではなく、新しいADRを追加し、古いADRの `Status` を `Superseded` に更新する。

## 保存場所と命名

ADRは `docs/adr/` 以下に置く。

ファイル名は以下の形式にする。

```text
0001-short-title.md
0002-short-title.md
```

番号は連番にする。タイトルは英小文字とハイフンを使い、内容が分かる短い名前にする。

## 書式

基本書式は以下を使う。

```markdown
# ADR-0001: Title

## Status

Accepted

## Context

判断が必要になった背景、制約、問題を書く。

## Decision

採用する方針を短く明確に書く。

## Consequences

この決定による影響を書く。良い影響だけでなく、制約やトレードオフも書く。

## Alternatives Considered

検討したが採用しなかった選択肢を書く。採用しなかった理由も短く書く。
```

## Status

`Status` には以下を使う。

- `Proposed`: 提案中
- `Accepted`: 採用済み
- `Deprecated`: 現在は推奨しないが、完全な置き換え先は未確定
- `Superseded`: 別ADRに置き換えられた

## Initial Records

- [ADR-0001: Use a self-managed orchestrator](0001-use-self-managed-orchestrator.md)
- [ADR-0002: Use Redmine REST API from Python instead of Redmine MCP](0002-use-redmine-rest-api-not-mcp.md)
- [ADR-0003: Keep README as the first-read entrypoint](0003-keep-readme-as-entrypoint.md)
- [ADR-0004: Evolve LangGraph to step-level execution state](0004-evolve-langgraph-to-step-execution.md)
- [ADR-0005: Use LangChain Tools and Agent Loop](0005-use-langchain-tools-and-agent-loop.md)
- [ADR-0006: Add Single Process Polling Daemon](0006-add-single-process-polling-daemon.md)
- [ADR-0007: Use Redmine Assignee Agent Profiles](0007-use-redmine-agent-profiles.md)
