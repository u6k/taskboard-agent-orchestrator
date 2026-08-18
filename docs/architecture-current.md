# 現在のアーキテクチャ

この文書は、現時点の実装責務を固定するためのメモである。目標アーキテクチャは [architecture-target.md](architecture-target.md) に記載する。

## 全体像

現在の実装は、Redmineをタスクボードとして使うCLIである。`run-once` は指定されたエージェントプロフィールで1件だけ処理して終了し、`run-daemon` は単一プロセス内で有効なプロフィールを順番にpollして継続処理する。CLIがプロフィールごとの依存オブジェクトを組み立て、Redmineから対象チケットを取得し、LangGraphでチケット単位の会話状態を保持しながら、TaskOrchestratorが計画と実行分岐を担当する。

```text
CLI
  -> run-once
    -> workflow.run_once()
    -> Redmineから対象チケット取得
    -> TicketConversationGraph.run(issue)
      -> LangGraph checkpointでチケット会話を保存・再開
      -> TaskOrchestrator.create_plan()
      -> LangGraphノードでstepを1件ずつ選択・実行
      -> TaskOrchestrator.execute_single_step()
    -> SkillEventをRedmine更新へ反映
  -> run-daemon
    -> daemon.run_daemon()
      -> workflow.run_once() を繰り返し呼ぶ
      -> 対象なしの場合だけpolling interval待機
```

## 主要コンポーネント

### `cli.py`

`taskboard-agent run-once` と `taskboard-agent run-daemon` のエントリポイントである。

- `.env` から共有設定、TOMLから担当者別の接続先、資格情報、LLMタイムアウトを含むエージェントプロフィールを読む
- プロフィールごとに `RedmineClient`, `LiteLLMClient`, `ChatLiteLLM`, `ToolScriptCatalog`, `TaskOrchestrator`, `TicketConversationGraph` を組み立てる
- `--dry-run` では `InMemorySaver` を使う
- 通常実行ではSQLite Checkpointerを使う
- `run-once` では `workflow.run_once()` を1回呼び、結果を標準出力へ表示する
- `run-daemon` では `daemon.run_daemon()` を呼び、1件処理後は待たずに次の検索を行う

### `daemon.py`

単一プロセスの常駐ループを担当する。

- 各巡回で有効な全プロフィールについて `workflow.run_once()` を1回ずつ呼ぶ
- `RunResult.status == "no_issue"` の場合だけ `interval_seconds` 秒待機する
- 全プロフィールが `no_issue` の場合だけ待機し、1件でも処理した場合は待機せず次の巡回へ進む
- `--max-iterations` は開発時やテスト時に指定できる
- `--dry-run` のdaemon実行では、同じチケットの再処理を避けるため `--max-iterations` を必須にする

### `workflow.py`

Redmineチケットの取得と、実行イベントのRedmine反映を担当する。

- `--issue-id` があればそのチケットを取得する
- 指定がなければ選択されたプロフィールのRedmine担当者に割り当てられた未完了チケットを1件取得する
- `SkillEvent` をRedmineコメント、ステータス、担当者更新へ変換する
- `start` で進行中にする
- `progress` でコメントを追加する
- `final_review` / `final_return` で起票者へ担当を戻し、レビュー中にする

Redmine操作はMCPではなく、Python実装からRedmine REST APIを呼び出す。

### `TicketConversationGraph`

LangGraphによるチケット単位の会話状態管理を担当する。
詳細なノード、エッジ、stateは [langgraph-ticket-graph.md](langgraph-ticket-graph.md) に記載する。

- `redmine-issue-{issue_id}` をthread IDにする
- 初回実行時にチケット本文と既存コメントをLangChain messageへ変換する
- 初回計画、初回実行、待機、差し戻し解析、再計画、再実行を状態遷移として扱う
- 人間の追加コメントだけを取り込み、保存済み会話へ追加する
- `select_next_step` / `execute_step` / `finalize_execution` でstepを1件ずつ実行する
- 各stepの `status`, `result`, `error`, `artifacts` をstateに保存する
- 実行結果のartifactを会話コンテキストとstateに保存する

現在のLangGraphは、会話履歴、再開状態、step単位の実行状態を保持する。stepは配列から削除せず、`pending`, `running`, `completed`, `failed`, `needs_user`, `skipped` のstatusで管理する。

### `TaskOrchestrator`

タスクの計画と実行分岐を担当する。

- `LiteLLMTaskPlanner` でチケット、利用可能スキル、利用可能toolから `TaskPlan` を作る
- `decision` に応じて `use_skill`, `use_tools`, `no_skill`, `needs_user` を分岐する
- `execute_single_step()` でLangGraphから1stepだけ実行できる部品を提供する
- `execute_plan()` と `_execute_steps()` は互換経路として残り、内部では `execute_single_step()` を呼ぶ
- LLM step、tool step、skill step、unavailable stepを処理する

現在の大きな構造的特徴は、通常のチケット実行では `TicketConversationGraph` がstep実行ループを持ち、`TaskOrchestrator` は1step実行の実作業を担当する点である。これにより、stepごとのcheckpointから未完了stepや停止stepを判定できる。

### LangChain Tool / `LangChainAgentRunner`

現在のtool定義とfunction calling loopはLangChain/LangGraph標準へ寄せている。

- `tool_scripts/{tool_name}.py` が `create_tool(context)` を公開し、LangChain `@tool`、型注釈、docstringでtool schemaを定義する
- `ToolScriptCatalog` が必要なtoolだけを読み込み、LangChain `BaseTool` を返す
- `LangChainAgentRunner` が `langchain.agents.create_agent()` を使い、tool call loopとtool実行をLangGraphベースのagent harnessへ委譲する
- `BaseTool.extras` の `risk` によってwrite toolや承認必須toolを制御する
- `--dry-run` では書き込みを抑止する

## 現在の制約

- step単位のRedmine進捗コメントは、既存の `progress` / `final_review` / `final_return` に寄せている。step開始、完了、スキップ、判断待ち、失敗は `progress` として記録する
- `completed_steps` は互換用に残っているが、step実行管理の中心は `plan_steps` と `step_results` である
- `TaskOrchestrator.execute_plan()` の互換経路は残っているため、完全な移行後に整理余地がある
- Redmine更新責務は `workflow.py` にあり、LangGraphノードから直接Redmine APIを更新しない

## 維持する前提

- Redmineは当面のタスクボードである
- Redmine操作はPythonからRedmine REST APIで行う
- `SkillEvent` からRedmineコメント・ステータス・担当者更新へ変換する責務は、当面 `workflow.py` に残す
- 危険操作や外部書き込みは、人間承認とdry-run方針が整理されるまで慎重に扱う
