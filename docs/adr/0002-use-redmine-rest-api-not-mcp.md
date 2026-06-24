# ADR-0002: Use Redmine REST API from Python Instead of Redmine MCP

## Status

Accepted

## Context

Redmineは、このプロジェクトで最初に使うタスクボードである。エージェントは、担当チケットの検索、チケット詳細と履歴の取得、コメント追加、ステータス更新、担当者変更を行う必要がある。

Redmine MCP連携を使う案も考えられた。しかし現在の実装には、Redmine APIを呼び出すPythonクライアントと `tool_scripts` が既にある。また、目指す制御モデルでは、タスクボード操作をオーケストレーターの管理境界内に置く必要がある。

## Decision

Redmine REST APIを呼び出すPythonコードでRedmineを操作する。

タスクボード連携の仕組みとしてRedmine MCPは導入しない。Redmine固有の操作は、`RedmineClient`、ワークフローコード、明示的なPython toolに残す。

## Consequences

オーケストレーターは、Redmineを変更する前に、dry-runの扱い、書き込みポリシー、ステータス遷移、担当者ルール、監査用の挙動を強制できる。

その代わり、このプロジェクトが保持する連携コードは増え、Redmine APIのリクエストとレスポンス処理を保守する必要がある。将来ほかのタスクボードを追加する場合は、Redmine MCPを特別な制御経路として導入するのではなく、タスクボードアダプターの抽象化を検討する。

## Alternatives Considered

- Redmine MCPを使う。
  - Redmine更新は、明示的なPythonワークフローとポリシー制御の下に置きたいため採用しない。
- Redmine操作を、LLMが選択する汎用toolの中だけに置く。
  - ステータス遷移や担当者変更は業務ルールであり、制約の弱いtool選択に依存させるべきではないため、主要なワークフロー更新では採用しない。
