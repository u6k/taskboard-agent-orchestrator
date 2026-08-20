# ADR-0008: Use Ticket Sessions and Content-addressed Artifacts

## Status

Accepted

## Context

Redmine差し戻し後の再計画とstep実行では、LangGraphに保存した会話履歴全体と長文成果物をLLM入力へ再展開していた。長期会話では同じ本文がcheckpointとpromptへ重複し、context上限、応答時間、SQLiteサイズを増大させる。一方、ユーザーは「先ほどの設定」「引き続き」のような通常の会話表現で、過去の成果物を参照できる必要がある。

## Decision

- Redmine issue IDをsession ID、journalをconversation turnとして扱う。
- Redmineを完全な会話台帳、LangGraph checkpointを実行状態とmodel-visible context、checkpoint DB隣接のartifact storeを長文成果物の正本とする。
- model-visible contextを`working_memory`、`session_checkpoint`、`recent_turns`、`active_artifacts`へ分離する。
- context window閾値を超える場合だけ、担当プロフィールと同じLLMで古いturnをStrict JSONへ圧縮する。
- 全assistant正規回答とstep/tool成果物をcontent-addressed JSONとして原子的に保存し、stateには`ArtifactRef`だけを保存する。dry-runはインメモリ保存にする。
- `depends_on`は同一TaskPlan内だけに限定し、複数turn間はsession contextとartifactで接続する。
- 旧checkpointは再開時に遅延変換し、一括移行・削除・VACUUMを行わない。
- LLM計測ログにはサイズ、所要時間、成否だけを含め、prompt、response、secret、artifact本文を含めない。

## Consequences

- 長期会話でも通常の自然言語参照を保ちつつ、毎回送る入力を抑えられる。
- artifactの選択、version、整合性検証が新しい永続化責務になる。
- compaction失敗時は履歴を削除せず安全に停止するため、一時的にユーザー確認または運用レビューが必要になる。
- 設定version 2への移行が必要で、全プロフィールに実際のcontext windowを明示しなければ起動できない。

## Alternatives Considered

- 最新journalだけをLLMへ渡す: 長期会話の決定事項と成果物参照を失うため不採用。
- 全履歴を毎回渡す: 現在のサイズ・速度問題を解消しないため不採用。
- Redmine本文だけを正本にする: step/toolの構造化成果物とversion管理に不向きなため不採用。
