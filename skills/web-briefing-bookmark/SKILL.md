---
name: "web-briefing-bookmark"
description: "指定URLのWebページ本文を取得し、ブリーフィング要約を作成してLinkAceへ登録する。"
required_tools:
  - linkace_check_auth
  - fetch_web_page
  - summarize_briefing
  - linkace_find_link
  - linkace_add_link
risk_level: "write"
---

# 手順

1. チケット本文から対象URLを1件特定する。
2. dry-runでなければ `linkace_check_auth` でLinkAce APIトークンの認証を確認する。
3. `linkace_find_link` で既存ブックマークを確認する。
4. 既存ブックマークがあり、tool結果の `bookmark.has_source_list` が false の場合は、要約や登録を行わず、既存ブックマークURLを示して `already_done` で終了する。
5. `fetch_web_page` でページタイトルと本文を取得する。
6. リダイレクトなどで最終URLが変わった場合は、最終URLでも `linkace_find_link` を実行する。既存ブックマークがあり `bookmark.has_source_list` が false の場合は、要約や登録を行わず、既存ブックマークURLを示して `already_done` で終了する。
7. `summarize_briefing` で客観的なブリーフィング要約を作成する。
8. `linkace_add_link` でURL、タイトル、要約を登録する。dry-runでもpayload確認のために呼び出す。
9. 最終応答JSONの `notes` に実施内容、成果物URL、確認してほしい点を記録する。

# 注意点

- LinkAce登録は書き込み操作なので、dry-run または承認ポリシーを必ず尊重する。
- ログイン、有料ページ、本文抽出不能、既存登録済みの場合は作業を中断し、人間が判断できるコメントを残す。
- 複数URLがある場合は依頼対象として最も明確な1件だけを処理する。
