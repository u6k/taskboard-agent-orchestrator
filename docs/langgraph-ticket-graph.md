# TicketConversationGraph のLangGraph構造

この文書は、現在の `TicketConversationGraph` が構築するLangGraphのノード、エッジ、state、再開動作を説明する。実装の正は `src/taskboard_agent/ticket_graph.py` であり、この文書は実装を読むための地図として扱う。

## 目的

`TicketConversationGraph` は、Redmineチケット1件をLangGraph threadとして扱い、会話履歴、実行計画、step単位の実行状態、差し戻し後の再計画をcheckpointへ保存する。

Redmine API更新はこのグラフから直接行わない。グラフは `SkillEvent` をemitし、`workflow.py` がRedmineコメント、ステータス、担当者変更へ変換する。

## グラフ全体

```mermaid
flowchart TD
    Start([START])
    Initialize["initialize<br/>チケット本文・コメントを会話stateへ初期化"]
    InitialPlan["initial_plan<br/>初回計画を作成"]
    PublishInitial["publish_initial_plan<br/>計画をstartイベントとして通知"]
    SelectStep{"select_next_step<br/>pending/running stepあり?"}
    ExecuteStep["execute_step<br/>1stepを実行して結果を保存"]
    MoreSteps{"未完了stepを継続できる?"}
    Finalize["finalize_execution<br/>実行全体の結果を確定"]
    Wait["wait_for_human<br/>interruptして人間コメント待ち"]
    Feedback{"人間の追加コメントあり?"}
    Analyze["analyze_feedback<br/>差し戻し内容を解析して再計画"]
    PublishRevision["publish_revision_plan<br/>再計画をstartイベントとして通知"]
    RequestFeedback["request_feedback<br/>修正指示不足をレビュー戻し"]

    Start --> Initialize --> InitialPlan --> PublishInitial --> SelectStep
    SelectStep -->|yes| ExecuteStep
    SelectStep -->|no| Finalize
    ExecuteStep --> MoreSteps
    MoreSteps -->|yes| SelectStep
    MoreSteps -->|no| Finalize
    Finalize --> Wait
    Wait --> Feedback
    Feedback -->|yes| Analyze --> PublishRevision --> SelectStep
    Feedback -->|no| RequestFeedback --> Wait
```

実装上、`select_next_step` と `execute_step` の分岐はLangGraphのconditional edgeで表現している。`MoreSteps` と `Feedback` は説明用の分岐ラベルであり、独立したノードではない。

## ノード責務

| ノード | 責務 | 主なstate更新 |
| --- | --- | --- |
| `initialize` | Redmine issue本文とjournalをLangChain messageへ変換し、初期stateを作る。設定済みエージェントユーザーのコメントは `AIMessage`、それ以外は `HumanMessage` として扱う。 | `issue_id`, `issue`, `messages`, `last_ingested_journal_id`, `plan_steps`, `step_results`, `run_status` |
| `initial_plan` | `TaskOrchestrator.create_plan()` で初回計画を作る。旧形式のstepなし計画は実行可能な単一stepへ補完する。 | `current_plan`, `plan_steps`, `current_step_index`, `step_context` |
| `publish_initial_plan` | 初回計画をRedmine向け `start` イベントとしてemitし、会話履歴にも残す。 | `messages` |
| `select_next_step` | `pending` または `running` のstepを1件選び、`running` にする。step開始を `progress` イベントとしてemitする。 | `plan_steps`, `current_step_index`, `run_status`, `messages` |
| `execute_step` | `TaskOrchestrator.execute_single_step()` に1stepだけ渡して実行する。結果、artifact、stepイベントをstateへ保存し、step結果を `progress` としてemitする。 | `plan_steps`, `step_results`, `current_step_index`, `run_status`, `step_context`, `artifacts`, `messages` |
| `finalize_execution` | 全stepの状態から実行全体の最終statusを決める。成功系は `progress` と `final_review`、失敗・判断待ちは `final_return` または `final_review` をemitする。 | `last_result`, `run_status`, `waiting_reason`, `current_step_index`, `step_context`, `messages` |
| `wait_for_human` | LangGraph `interrupt()` で停止し、次回実行時のresume payloadから新しいRedmine journalを取り込む。 | `issue`, `messages`, `last_ingested_journal_id`, `has_human_feedback`, `feedback_analysis` |
| `analyze_feedback` | 人間コメントをLLMで解析し、維持する成果、やり直す作業、再計画を構造化する。 | `feedback_analysis`, `current_plan`, `plan_steps`, `current_step_index`, `step_context` |
| `publish_revision_plan` | 差し戻し解析結果と再計画を `start` イベントとしてemitし、会話履歴に残す。 | `messages` |
| `request_feedback` | resume時に人間の追加指示が確認できない場合、修正内容の追記を求めてレビュー戻しにする。 | `last_result`, `waiting_reason`, `run_status`, `messages` |

## 主要state

| state | 内容 |
| --- | --- |
| `issue_id` | Redmine issue ID。thread IDは `redmine-issue-{issue_id}` を使う。dry-runでは `-dry-run` suffixを付ける。 |
| `issue` | journalを除いたissue情報。resume時は最新issueで更新する。 |
| `messages` | チケット本文、Redmineコメント、計画、実行結果を保持する会話履歴。LangGraphの `add_messages` reducerで追加される。 |
| `last_ingested_journal_id` | 取り込み済みRedmine journalの最大ID。resume時に重複取り込みを避ける。 |
| `current_plan` | 現在実行中の `TaskPlan` をdict化したもの。 |
| `plan_steps` | 実行計画のstep配列。stepは削除せず `status` を更新する。 |
| `current_step_index` | 現在実行中または停止中のstep index。成功完了時は `None` に戻す。 |
| `step_results` | 実行済みstepごとの結果履歴。step ID、status、実行結果、イベント、artifactを保存する。 |
| `run_status` | グラフ全体の現在status。例: `initialized`, `planned`, `running`, `processed`, `failed`, `needs_user`, `missing_tool`, `dry_run`。 |
| `step_context` | step実行間で共有する会話文脈、計画理由、制約、最後のstatusなど。 |
| `artifacts` | step実行で得た成果物メタデータ。 |
| `feedback_analysis` | 差し戻しコメントを解析した構造化結果。 |
| `waiting_reason` | `wait_for_human` で停止する理由。 |
| `last_result` | `TicketConversationGraph.run()` が返す最終 `SkillExecutionResult` 相当のdict。 |

## Step状態

`plan_steps` の各stepは配列から削除せず、以下のstatusで管理する。

| status | 意味 |
| --- | --- |
| `pending` | 未実行。 |
| `running` | `select_next_step` で選択済み。 |
| `completed` | `processed`, `already_done`, `dry_run` として完了。 |
| `skipped` | 実行対象外としてスキップ。 |
| `needs_user` | 人間判断待ち。`missing_tool` もstep状態としてはここに寄せる。 |
| `failed` | step実行に失敗。 |

最終statusは `finalize_execution` で `plan_steps` を走査して決める。`failed` があれば全体は `failed`、`needs_user` があれば `needs_user` または `missing_tool`、それ以外は `processed` または `dry_run` になる。

## Redmine向けイベント

グラフはRedmineを直接更新せず、以下の `SkillEvent` をemitする。

| タイミング | kind | 内容 |
| --- | --- | --- |
| 初回計画公開 | `start` | 計画内容と作業開始。`workflow.py` で進行中ステータスへ更新される。 |
| step開始 | `progress` | `ステップ N を開始しました: ...` |
| step実行中の詳細 | `progress` | `TaskOrchestrator.execute_single_step()` から返る進捗や結果。 |
| step結果 | `progress` | 完了、スキップ、判断待ち、失敗を記録する。 |
| 全体成功 | `progress` + `final_review` | step全体の終了メッセージをprogressにし、最後にレビュー戻しする。 |
| 全体失敗・判断待ち | `final_return` または `final_review` | 既存の担当者戻し、レビュー中ステータス更新仕様を維持する。 |
| 差し戻し後の再計画 | `start` | 差し戻し内容、維持する成果、今回対応する内容、再計画。 |

## 再開動作

`run()` はcheckpointの状態を見て、次の3経路に分岐する。

1. checkpointがなく初回実行の場合  
   `initialize` から開始する。

2. `wait_for_human` でinterrupt中の場合  
   最新issueから未取り込みjournalだけをresume payloadとして渡す。設定済みエージェントが過去に投稿した同一内容は重複取り込みしない。人間コメントがあれば `analyze_feedback` へ進む。

3. checkpointに `next` が残っている場合  
   前回プロセスが途中で止まった状態として、次ノードから継続する。差し戻し後のstep実行途中なら、再開コメントを `start` としてemitしてから続ける。

## 責務境界

- `TicketConversationGraph` は状態遷移、checkpoint、step選択、resume判定を担当する。
- `TaskOrchestrator` は計画作成と1step実行の実作業を担当する。
- `workflow.py` は `SkillEvent` をRedmine更新へ変換する。
- Redmine更新責務をLangGraphノードへ直接移さない。
