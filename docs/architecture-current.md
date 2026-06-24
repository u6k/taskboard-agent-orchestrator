# 現在のアーキテクチャ

この文書は、現時点の実装責務を固定するためのメモである。目標アーキテクチャは [architecture-target.md](architecture-target.md) に記載する。

## 全体像

現在の実装は、Redmineをタスクボードとして使う one-shot CLI である。CLIが依存オブジェクトを組み立て、Redmineから対象チケットを取得し、LangGraphでチケット単位の会話状態を保持しながら、TaskOrchestratorが計画と実行分岐を担当する。

```text
CLI
  -> workflow.run_once()
    -> Redmineから対象チケット取得
    -> TicketConversationGraph.run(issue)
      -> LangGraph checkpointでチケット会話を保存・再開
      -> TaskOrchestrator.create_plan()
      -> TaskOrchestrator.execute_plan()
    -> SkillEventをRedmine更新へ反映
```

## 主要コンポーネント

### `cli.py`

`taskboard-agent run-once` のエントリポイントである。

- `.env` と環境変数から設定を読む
- `RedmineClient`, `LiteLLMClient`, `SkillRegistry`, `ToolScriptCatalog`, `TaskOrchestrator`, `TicketConversationGraph` を組み立てる
- `--dry-run` では `InMemorySaver` を使う
- 通常実行ではSQLite Checkpointerを使う

### `workflow.py`

Redmineチケットの取得と、実行イベントのRedmine反映を担当する。

- `--issue-id` があればそのチケットを取得する
- 指定がなければAIユーザーに割り当てられた未完了チケットを1件取得する
- `SkillEvent` をRedmineコメント、ステータス、担当者更新へ変換する
- `start` で進行中にする
- `progress` でコメントを追加する
- `final_review` / `final_return` で起票者へ担当を戻し、レビュー中にする

Redmine操作はMCPではなく、Python実装からRedmine REST APIを呼び出す。

### `TicketConversationGraph`

LangGraphによるチケット単位の会話状態管理を担当する。

- `redmine-issue-{issue_id}` をthread IDにする
- 初回実行時にチケット本文と既存コメントをLangChain messageへ変換する
- 初回計画、初回実行、待機、差し戻し解析、再計画、再実行を状態遷移として扱う
- 人間の追加コメントだけを取り込み、保存済み会話へ追加する
- 実行結果のartifactをstateに保存する

現在のLangGraphは、主に会話履歴と再開状態の保持に使われている。作業stepそのものは、LangGraphノードとして1件ずつ実行されていない。

### `TaskOrchestrator`

タスクの計画と実行分岐を担当する。

- `LiteLLMTaskPlanner` でチケット、利用可能スキル、利用可能toolから `TaskPlan` を作る
- `decision` に応じて `use_skill`, `use_tools`, `no_skill`, `needs_user` を分岐する
- `steps` がある場合は `_execute_steps()` でPythonのforループとして順番に実行する
- LLM step、tool step、skill step、unavailable stepを処理する

現在の大きな構造的特徴は、`TaskOrchestrator._execute_steps()` がstep列をまとめて実行している点である。LangGraphから見ると、step実行の途中経過は1つのノード内部の処理になっている。

### `ToolRegistry` / `FunctionCallingAgent`

現在のtool定義とfunction calling loopは自前実装である。

- `tool_scripts/{tool_name}.py` が `TOOL_SPEC` と `create_handler(context)` を公開する
- `ToolScriptCatalog` が必要なtoolだけを読み込み、`ToolRegistry` を組み立てる
- `FunctionCallingAgent` がLiteLLMにtool schemaを渡し、tool callを受け取り、`ToolRegistry.execute()` で実行する
- `risk` によってwrite toolや承認必須toolを制御する
- `--dry-run` では書き込みを抑止する

## 現在の制約

- stepごとの状態はLangGraph stateとして十分に表現されていない
- step完了ごとのcheckpointではなく、まとまった実行結果として保存されやすい
- `completed_steps` は存在するが、step実行管理の中心にはなっていない
- LangGraphの `ToolNode` やLangChain Toolへはまだ寄せていない
- Redmine更新責務は `workflow.py` にあり、LangGraphノードから直接Redmine APIを更新しない

## 維持する前提

- Redmineは当面のタスクボードである
- Redmine操作はPythonからRedmine REST APIで行う
- `SkillEvent` からRedmineコメント・ステータス・担当者更新へ変換する責務は、当面 `workflow.py` に残す
- 危険操作や外部書き込みは、人間承認とdry-run方針が整理されるまで慎重に扱う
