---
name: "weekly-docx-report-extractor"
description: "Redmineチケットに添付された日本語の週報DOCXを1件ずつ解析し、案件進捗、障害・ネガティブ情報、営業情報、自由意見の管理職向けサマリーをチケットコメントへ出力する。"
runner: "run.py"
required_tools:
  - extract_redmine_docx
  - summarize_weekly_docx
risk_level: "read"
---

# 目的

Redmineチケットの全DOCX添付を週報として処理し、ファイルごとのサマリーをコメントする。
この文書をワークフロー仕様の正本とし、添付列挙、処理順序、失敗制御、イベント種別は `run.py` で決定的に実行する。

# 作業手順

1. チケットの `attachments` から、ファイル名が `.docx` で終わる添付を記載順に抽出する。
2. 各添付について `extract_redmine_docx` を呼び、段落と入れ子表を文書順に復元する。
3. 復元本文を `summarize_weekly_docx` へ渡し、LLMに内容の抽出、分析、グルーピング、要約を行わせる。
4. 各ファイルの要約を、それぞれ独立した `progress` イベントとして返す。
5. 全ファイルの処理後、対象数、成功数、失敗数、処理ファイル一覧を `final_review` イベントとして返す。

# 失敗時の扱い

- DOCX添付がない場合は、添付を依頼する `final_review` を返して終了する。
- 個別ファイルの取得、抽出、要約に失敗しても後続ファイルを処理する。
- 失敗したファイルにも、ファイル名、失敗工程、理由を記載した `progress` イベントを1件返す。
- 1件以上失敗した場合の状態は `needs_user`、全件成功は `processed` とする。
- dry-runでは同じ読取・要約処理を行い、Redmineへ書き込まず状態を `dry_run` とする。

# 出力規則

- ファイル単位のコメントはすべて `progress` とする。
- `final_review` には個別サマリーを重複掲載しない。
- 非DOCX添付は無視し、対象数に含めない。
- Markdownファイルの保存やDOCXのコピーは行わない。
