# ADR-0007: Use Redmine Assignee Agent Profiles

## Status

Accepted

## Context

単一の `REDMINE_AI_USER_ID` と `LLM_MODEL` では、Redmine担当者ごとに異なる役割、モデル、接続先、資格情報を使い分けられない。複数プロフィールを同時実行すると、現在の単一イベントsinkとSQLite checkpointの前提も見直す必要がある。

## Decision

エージェント固有設定をTOMLのプロフィールとして定義し、RedmineユーザーID、Redmine APIキー、LLMモデル、API endpoint、API key、任意のsystem promptファイルを保持する。

1つのdaemonが有効なプロフィールを設定順に1件ずつ巡回する。プロセス内並列実行は行わない。LangGraph threadはissue単位のまま共有し、設定済みの全RedmineエージェントユーザーをAI投稿者として扱う。

## Consequences

- 担当者ごとにRedmine名義とLLM接続先を分離できる。
- 1プロフィールの長時間タスク中は、後続プロフィールの処理開始が待たされる。
- APIキーを平文TOMLで管理するため、将来SecretVaultへ取得元を差し替える必要がある。
- 単一エージェント用の旧環境変数との後方互換性は持たない。

## Alternatives Considered

- 担当者ごとにdaemonプロセスを分ける。
  - 並列化できるが、初期導入でプロセス管理とcheckpoint競合の運用が増えるため採用しない。
- 1プロセス内でプロフィールを並列実行する。
  - 現在のイベントsinkとSQLite利用を同時実行対応へ変更する必要があり、今回の変更範囲を超えるため採用しない。
- 環境変数を担当者数だけ増やす。
  - 構造化された検証とプロフィール追加が難しくなるため採用しない。
