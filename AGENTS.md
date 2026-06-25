# エージェント向け開発方針

このファイルは、Codexなどの開発エージェントが本リポジトリで作業するときの判断基準である。実装前に必ず読むこと。

## 最初に読む文書

1. `README.md`
   - プロジェクトの目的、背景、セットアップ、実行方法を把握する。
2. `docs/architecture-current.md`
   - 現在の責務分担と実装構造を確認する。
3. `docs/architecture-target.md`
   - 目標アーキテクチャと採用しない方針を確認する。
4. `docs/roadmap.md`
   - 段階的な改善順序を確認する。
5. `docs/adr/README.md`
   - 採用済みのアーキテクチャ判断と、ADR追記ルールを確認する。
6. `docs/tools-and-skills.md`
   - 現在利用できるtoolとskillを確認する。
7. `docs/use-cases.md`
   - 実装対象ユースケースと成長順を確認する。

READMEは初回読者向けの入口である。詳細設計、内部構造、改修計画は `docs/` 以下に書く。

## 基本方針

- 自前オーケストレーターで業務制御を持つ。
- LangGraphを、チケット単位の会話保存だけでなく、step単位の実行状態機械へ発展させる。
- Redmineは当面のタスクボードとして扱う。
- Redmine操作はMCPではなく、Python実装からRedmine REST APIで行う。
- PDF抽出は現時点の要件ではない。必要になった場合は別途要件化して設計する。
- OpenClawを実行基盤として採用しない。

## 変更の進め方

- 1回の変更で複数Phaseをまとめて実装しない。
- 既存挙動を維持しながら、小さく分解して進める。
- 仕様変更を伴う場合は、先に `docs/` を更新する。
- アーキテクチャ上の重要判断を変更または追加する場合は、`docs/adr/` にADRを追加する。
- READMEに詳細設計を戻さない。
- 新しい機能や状態遷移を追加する場合は、対応するテストを追加する。
- 既存のRedmineコメント、ステータス更新、担当者戻しの仕様を不用意に変えない。
- エラー対応では、目の前の入力や特定チケットだけを特別扱いする修正を避け、同種の失敗に再利用できる汎用的な検証、補正、リトライ、フォールバックを優先する。個別対応が必要な場合も、なぜ汎用化できないかを明確にする。

## LangGraph移行方針

現在は、`TicketConversationGraph` がLangGraphで会話と再開状態を保持し、`TaskOrchestrator._execute_steps()` がPythonのforループでstep列をまとめて実行している。

目標は、step列の実行をLangGraphノード遷移へ移すことである。

優先順:

1. LangGraph stateにstep状態を追加する。
2. `TaskOrchestrator` から1step実行関数を切り出す。
3. LangGraphノードで1stepずつ実行する。
4. step単位の進捗、失敗、再開をRedmineコメントとcheckpointへ反映する。
5. tool定義とfunction calling loopはLangChain/LangGraph標準を使い、業務policyは自前制御に残す。

stepは配列から削除せず、`pending`, `running`, `completed`, `failed`, `needs_user`, `skipped` のようなstatusで管理する。

## Redmine連携方針

- Redmine APIクライアントと `tool_scripts` を使って操作する。
- Redmine MCPは導入しない。
- `workflow.py` の `SkillEvent -> Redmine更新` の責務は当面維持する。
- step単位実行へ移行しても、最初は既存の `start`, `progress`, `final_review`, `final_return` を維持する。
- write操作、ステータス変更、担当者変更はdry-runと承認方針を必ず考慮する。

## Tool / Skill方針

- `tool_scripts/{tool_name}.py` は `create_tool(context)` を公開し、LangChain `@tool`、型注釈、docstringを正式なtool定義とする。
- スキルは `skills/{skill_name}/SKILL.md` を正式な手順書として扱う。
- `runner: "run.py"` があるスキルは、決定的なPython runnerを優先する。
- LLM function callingに依存しすぎず、確実に実行したい手順はscripted runnerへ寄せる。
- LangChain ToolやLangGraph agent loopを使う場合も、tool policy、dry-run、write禁止、人間承認は自前制御を維持する。

## テスト方針

- 実装変更後は原則として `uv run pytest` を実行する。
- ドキュメントのみの変更ではユニットテストは不要だが、リンク先や方針の矛盾を目視確認する。
- 状態遷移、再開、差し戻し、Redmine更新に関わる変更では、既存テストに加えて対象ケースのテストを追加する。
- 外部サービスを実際に更新するテストは避け、fake clientまたはdry-runで検証する。

## 禁止事項

- READMEを詳細設計書として肥大化させること。
- Redmine MCPを前提に実装や文書を書くこと。
- PDF抽出を暗黙の要件として追加すること。
- `TaskOrchestrator._execute_steps()` の大改修とLangGraphノード移行を同じ変更でまとめて行うこと。
- 既存のユーザー変更や無関係な差分を巻き戻すこと。
