# 目標アーキテクチャ

この文書は、taskboard-agent-orchestrator が目指す構造を定義する。実装の現在地は [architecture-current.md](architecture-current.md) に記載する。

## 基本方針

本プロジェクトは、OpenClawのような外部実行基盤に作業制御を委ねるのではなく、自前オーケストレーターで業務制御を持つ。LangGraphは、チケット単位の会話保存だけでなく、作業step単位の実行状態機械として使う。

Redmineは初期のタスクボードであり、操作はMCPではなくPython実装からRedmine REST APIで行う。将来ほかのタスクボードへ対応する場合は、Redmine前提の業務ルールを直接広げるのではなく、Taskboard adapterとして抽象化する。

## 目標とする責務分担

```text
Taskboard Agent Orchestrator
  -> Workflow / Scheduler
    -> Redmine担当者別プロフィールを順番に巡回する
    -> タスクボードから処理対象を取得
    -> 実行イベントをタスクボードへ反映
  -> TicketConversationGraph
    -> LangGraph thread/checkpoint
    -> plan_stepsとstep結果の状態管理
    -> 初回実行、差し戻し、再開の状態遷移
  -> TaskOrchestrator
    -> 計画作成
    -> 1step実行の部品提供
    -> skill/tool/llmの実行分岐
  -> Tool / Skill Runtime
    -> 実作業
    -> 外部API、Web取得、ファイル処理、ブックマーク登録
```

`workflow.py` はタスクボードI/Oを担当し、`TicketConversationGraph` は実行状態を管理し、`TaskOrchestrator` は1stepを実行する部品を提供する。最終的には、step列のループを `TaskOrchestrator` 内のforループではなく、LangGraphのノード遷移として表現する。

エージェントプロフィールはRedmine担当者、Redmine APIキー、LLMモデル、endpoint、LLM APIキー、任意のsystem promptを束ねる。初期の複数エージェント実行は単一daemonによる順次巡回とし、同時実行は別要件として扱う。

## 目標フロー

```text
initialize
  -> plan
  -> publish_plan
  -> select_next_step
  -> execute_step
  -> record_step_result
  -> route_after_step
      -> pending stepあり: select_next_step
      -> all completed: finalize_success
      -> failed: finalize_failed
      -> needs_user: wait_for_human
  -> wait_for_human
  -> analyze_feedback
  -> plan revision
```

この構造では、LangGraph stateに計画、現在step、step結果、artifact、待機理由が残る。途中でプロセスが停止しても、どのstepまで終わったか、どのstepから再開すべきかをcheckpointから判断できる。

## Step状態

stepは配列から削除せず、statusを更新して監査可能にする。

想定するstep状態:

- `pending`: 未実行
- `running`: 実行中
- `completed`: 完了
- `failed`: 失敗
- `needs_user`: 人間判断待ち
- `skipped`: 再計画などにより実行しない

各stepには少なくとも以下を保持する。

- `id`
- `index`
- `kind`
- `name`
- `purpose`
- `arguments`
- `status`
- `result`
- `error`
- `artifacts`

## Redmine連携方針

当面、RedmineはPython APIクライアントと `tool_scripts` から操作する。Redmine MCPは採用しない。

Redmine更新は以下の方針で扱う。

- AI担当チケットの検索、チケット取得、コメント追加、ステータス変更、担当者変更はRedmine REST APIで行う
- Redmineへの進捗反映は `SkillEvent` を通じて行う
- step単位実行へ移行しても、最初は既存の `start`, `progress`, `final_review`, `final_return` を維持する
- 将来 `step_started`, `step_completed`, `step_failed` のようなイベントを追加する場合も、Redmine更新責務の所在を明確にする

## Tool / Skill方針

tool定義とfunction calling loopはLangChain ToolとLangGraphベースのagent harnessへ寄せる。ただし、業務上の制御は自前オーケストレーター側に残す。

自前で維持するもの:

- このチケットで使ってよいtoolの選択
- write toolの許可
- `--dry-run` 時の書き込み禁止
- Redmineステータス遷移
- 担当者変更ルール
- 人間承認が必要な操作の停止
- 監査ログ
- 再開時のstep判定

LangChain / LangGraphへ寄せるもの:

- tool schema生成
- function calling loop
- tool実行エラーの標準化
- `ToolNode` 相当のtool実行

Redmine更新、外部書き込み、承認が必要な操作は、LangChain Toolとして表現しても自前policyを優先する。

## 対象外

以下は現時点の目標アーキテクチャには含めない。

- PDF抽出機能の新規要件化
- Redmine MCPの導入
- OpenClawを実行基盤として採用すること
- READMEへの詳細設計の集約

これらが必要になった場合は、別の要件として設計する。
