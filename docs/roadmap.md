# 段階的改善ロードマップ

このロードマップは、現在の実装をLangGraph中心のstep単位実行モデルへ移行するための順序を示す。各Phaseは小さく実装し、既存挙動を壊さないことを優先する。

## Phase 0: ドキュメント整理

目的:

- READMEを初回読者向けに整理する
- 現在の実装構造を文書化する
- 目標アーキテクチャを文書化する
- エージェント向け開発方針を `AGENTS.md` に記載する
- 主要なアーキテクチャ判断をADRとして `docs/adr/` に記録する

完了条件:

- `README.md` が目的、背景、使い方、docs導線に絞られている
- `docs/architecture-current.md` が現在の責務分担を説明している
- `docs/architecture-target.md` がLangGraph step単位実行モデルを説明している
- `AGENTS.md` が後続エージェントの判断基準になっている
- `docs/adr/` に採用済みの主要判断が記録されている

## Phase 1: LangGraph stateにstep状態を追加する

目的:

- 既存挙動を変えずに、LangGraph state上へstep単位の状態を持てるようにする

候補state:

- `plan_steps`
- `current_step_index`
- `step_results`
- `run_status`
- `step_context`

方針:

- `current_plan` から `plan_steps` を作れるようにする
- まだ `execute_initial -> TaskOrchestrator.execute_plan()` の外部挙動は変えない
- stepは削除せず、status更新で管理する

完了条件:

- 初回計画後にLangGraph stateでstep列を確認できる
- 既存テストが通る
- 新しいstate変換のテストがある

## Phase 2: 1step実行関数を切り出す

目的:

- `TaskOrchestrator._execute_steps()` の巨大なforループを分解し、LangGraphから1stepだけ実行できる入口を作る

方針:

- `execute_single_step()` または `_execute_single_step()` を追加する
- 既存の `_execute_steps()` は新しい1step実行関数をforループ内で呼ぶ
- このPhaseでは外部挙動を変えない

完了条件:

- 1step実行単位のテストがある
- `_execute_steps()` の既存テストが通る
- skill/tool/llm/unavailable stepの挙動が維持されている

## Phase 3: LangGraphへstep実行ループを移す

目的:

- step列の実行を `TaskOrchestrator` 内のforループからLangGraphノード遷移へ移す

候補ノード:

- `prepare_steps`
- `select_next_step`
- `execute_step`
- `record_step_result`
- `route_after_step`
- `finalize_success`
- `finalize_failed`

方針:

- `execute_initial` / `execute_revision` が `TaskOrchestrator.execute_plan()` に丸投げする構造をやめる
- 1step実行ごとにLangGraph stateを更新する
- 失敗時や人間判断待ち時に、どのstepで止まったかstateから判定できるようにする

完了条件:

- stepごとに `status`, `result`, `error`, `artifacts` が保存される
- checkpointから未完了stepを判定できる
- 差し戻し時に完了済みstepとやり直し対象stepを区別できる

実装状況:

- `TicketConversationGraph` が `select_next_step` / `execute_step` / `finalize_execution` でstepを1件ずつ実行する
- `TaskOrchestrator.execute_single_step()` をLangGraphノードから呼び、`execute_plan()` への丸投げ経路を通常実行から外している
- `plan_steps` と `step_results` にstepごとの実行結果を保存する
- `TaskPlan.steps` が空の旧形式計画は、互換のため実行可能な単一stepへ補完する

## Phase 4: step単位の進捗反映を整える

目的:

- step単位の実行状態をRedmineコメントとcheckpointの両方で追えるようにする

方針:

- 最初は既存の `SkillEvent` kindを維持する
- step開始、step完了、step失敗は `progress` として出してよい
- `final_review` / `final_return` のRedmine更新仕様は既存どおり維持する
- 必要になった時点でイベント体系を拡張する

完了条件:

- Redmineコメントから進捗が追える
- LangGraph stateから機械的にstep状態が追える
- 途中失敗時の再開判断が明確になる

実装状況:

- `TicketConversationGraph` がstep選択時に `progress` としてstep開始コメントを出す
- step実行後に `processed` / `dry_run` / `already_done` は完了、`skipped` はスキップ、`needs_user` / `missing_tool` は判断待ち、その他は失敗として `progress` コメントを出す
- `plan_steps` と `step_results` にstepごとの状態、結果、artifactを保存する
- `final_review` / `final_return` によるレビュー戻しのRedmine更新仕様は維持する

## Phase 5: tool管理とfunction calling loopを標準化する

目的:

- tool定義、schema生成、function calling loopをLangChain/LangGraph標準へ寄せる

方針:

- `tool_scripts/{tool_name}.py` は `create_tool(context)` からLangChain `BaseTool` を返す
- tool schemaは `@tool`、型注釈、docstringから生成する
- LLMのtool call loopはLangChain `create_agent()` に委譲する
- Redmine更新、外部書き込み、承認が必要なtoolは自前policyを維持する
- tool catalogとtool policyは業務制御レイヤーとして残す

完了条件:

- 既存toolがLangChain `BaseTool` として定義されている
- 汎用tool/skill実行のfunction calling loopがLangChain/LangGraph agent harnessで動く
- dry-run、write禁止、人間承認の制御が壊れていない
- 既存スキル実行が維持されている

## 実装時の原則

- 1回の変更で複数Phaseをまとめて実装しない
- READMEへ詳細設計を書き戻さない
- RedmineはMCPではなくPython API操作を前提にする
- PDF抽出は現時点では要件化しない
- 既存テストを通し、状態変更を伴うPhaseではテストを追加する
