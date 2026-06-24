# Tool and Skill Catalog

この文書は、現在このリポジトリで利用できるtoolとskillを説明する。実装上の定義は `tool_scripts/` と `skills/` が正であり、この文書は読み手が全体像を把握するための補助である。

## 現在の実行能力

現在のCLIは、RedmineでAIユーザーが担当している未完了チケットを1件取得し、チケット本文とコメント履歴を読んで、利用可能なskillまたはtoolを選択して実行する。

実行中の計画、作業ログ、成果、差し戻しコメントはLangGraph checkpointに保存される。人間から差し戻しコメントが追加された場合は、保存済み会話に未取り込みコメントを追加し、再計画して必要な作業を再開する。

Redmineには進捗コメントを投稿し、作業後は起票者へ担当を戻してレビュー中にする。

## Skills

skillは、特定の業務を実行するための手順書である。`SKILL.md` のfront matterに名前、説明、必要tool、risk、runnerを定義する。`runner: "run.py"` があるskillは、LLMの自由実行ではなくPython runnerで決定的に実行する。

| Skill | Risk | Runner | 概要 | Required tools |
| --- | --- | --- | --- | --- |
| `web-briefing-bookmark` | `write` | `run.py` | 指定URLのWebページ本文を取得し、ブリーフィング要約を作成してLinkAceへ登録または更新する。 | `linkace_find_link`, `fetch_web_page`, `summarize_briefing`, `linkace_add_link` |
| `weekly-docx-report-extractor` | `read` | `run.py` | Redmineチケットに添付された日本語の週報DOCXを解析し、案件進捗、障害・ネガティブ情報、営業情報、自由意見を管理職向けに要約する。 | `extract_redmine_docx`, `summarize_weekly_docx` |

## Tools

toolは `tool_scripts/{tool_name}.py` として定義する。各toolは `create_tool(context)` を公開し、LangChain `@tool`、型注釈、docstringからtool schemaと説明を生成する。実行時に必要なサービスは `ToolRuntimeContext` から受け取り、`risk`, `planner_visible`, `dry_run_safe` は `BaseTool.extras` に保持する。

| Tool | Risk | Planner visible | 概要 |
| --- | --- | --- | --- |
| `fetch_web_page` | `read` | yes | 指定URLのWebページから最終URL、タイトル、本文を抽出する。 |
| `web_search_pages` | `read` | yes | DuckDuckGoで検索し、検索結果と各ページ本文の取得結果を返す。 |
| `summarize_briefing` | `read` | yes | Webページ本文からブリーフィング要約を生成する。 |
| `linkace_check_auth` | `read` | yes | LinkAce API tokenの認証状態を確認する。 |
| `linkace_find_link` | `read` | yes | 指定URLがLinkAceに登録済みか確認する。 |
| `linkace_add_link` | `write` | yes | URL、タイトル、ブリーフィング要約をLinkAceへ登録する。dry-runではpayloadだけ返す。 |
| `redmine_add_comment` | `write` | no | Redmineチケットへコメントを追加する。通常のワークフロー更新とは別の明示tool。 |
| `extract_redmine_docx` | `read` | no | RedmineのDOCX添付を取得し、段落と入れ子表を文書順のテキストへ復元する。 |
| `summarize_weekly_docx` | `read` | no | DOCXから抽出された週報本文をLLMで分析し、管理職向けMarkdownサマリーを生成する。 |

`Planner visible` が `no` のtoolは、通常のタスク計画時にLLMへ直接提示しない。主にskill runner内部など、決められた手順から呼び出すために使う。

## Risk Levels

- `read`: 外部状態を変更しない読み取り、取得、要約。
- `write`: Redmine、LinkAceなど外部サービスへ書き込む可能性がある操作。
- `approval_required`: 人間承認が必要な操作。現在のtool一覧では未使用。

`--dry-run` ではRedmineや外部サービスへの書き込みを抑止する。write toolのうちdry-run safeなものは、実際に書き込まず予定payloadを返す。

## Redmine操作

Redmine操作はMCPではなく、Python実装からRedmine REST APIを呼び出す。

通常のチケット取得、ステータス変更、担当者変更、進捗コメント投稿は `workflow.py` と `RedmineClient` の責務である。`redmine_add_comment` はtoolとしてコメント追加が必要な場合のために存在するが、通常の作業終了時のRedmine更新経路ではない。

## 対象外

PDF抽出は現時点の要件ではない。必要になった場合は、別途要件化し、toolまたはskillとして設計する。
