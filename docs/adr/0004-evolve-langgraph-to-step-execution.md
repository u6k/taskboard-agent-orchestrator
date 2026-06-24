# ADR-0004: Evolve LangGraph to Step-Level Execution State

## Status

Accepted

## Context

現在の実装では、LangGraphをチケット単位の会話状態、チェックポイント、フィードバック取り込み、修正フローに使っている。しかし、実際のタスクステップはまだ `TaskOrchestrator._execute_steps()` のPython forループ内で実行されている。

つまり、LangGraphから見えるのは大きな実行ノードであり、各ステップがチェックポイント付きの状態遷移としては見えていない。長時間実行や複数ステップの作業では、どのステップが未着手、実行中、完了、失敗、スキップ、人間待ちなのかをより明確に見えるようにする必要がある。

## Decision

LangGraphを、粗いチケット会話のラッパーから、ステップ単位の実行状態機械へ発展させる。

目標とするグラフでは、計画済みステップを状態として保持し、次のようなノードを通して実行を進める。

- `plan`
- `publish_plan`
- `select_next_step`
- `execute_step`
- `record_step_result`
- `route_after_step`
- `finalize_success`
- `finalize_failed`
- `wait_for_human`

`TaskOrchestrator` には計画作成と1ステップ実行のヘルパーを残す。ただし最終的には、ステップのループとステップ結果のチェックポイントはLangGraphが担う。

## Consequences

システムは、個別ステップごとにチェックポイントを作成し、グラフ状態から進捗を確認し、未完了の作業から再開し、特定の完了ステップや失敗ステップに対する人間のフィードバックを扱えるようになる。

移行は段階的に行う必要がある。まずステップ状態を追加し、次に1ステップ実行を切り出し、その後にループをLangGraphへ移す。一度に大きく書き換えると、既存のRedmine更新や修正フローを壊しやすい。

## Alternatives Considered

- ステップ実行をすべて `TaskOrchestrator._execute_steps()` の中に残す。
  - LangGraphから進捗が見えにくく、再開可能性も制限されるため採用しない。
- すべてのtool実行をすぐにLangGraphの `ToolNode` へ移す。
  - まず安定したステップ状態と業務ポリシー境界が必要であるため採用しない。ToolNodeへの移行は、低リスクな読み取りtoolから後で評価する。
