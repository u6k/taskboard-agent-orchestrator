# Agent Profile Configuration

この文書は、Redmine担当者ごとに異なるAIエージェントを実行するための設定と巡回規則を定義する。

## 設定ファイル

エージェント固有の設定はTOMLファイルで管理する。CLIは既定で `agents.toml` を読み、グローバルオプション `--config` で別のパスを指定できる。

```toml
version = 1

[[agents]]
id = "research-agent"
enabled = true
redmine_user_id = 123
redmine_api_key = "redmine-api-key-for-research-agent"
llm_model = "openai/gpt-4.1-mini"
llm_api_base = "https://api.openai.com/v1"
llm_api_key = "llm-api-key-for-research-agent"
llm_timeout_seconds = 1200
system_prompt_file = "agent-prompts/research-agent.md"
```

各プロフィールの項目:

- `id`: CLIとログで使う一意な識別子。
- `enabled`: 省略時は `true`。`false` のプロフィールは実行対象外。
- `redmine_user_id`: 処理対象となるRedmine担当者ID。
- `redmine_api_key`: この担当者名義でRedmineを読み書きするAPIキー。
- `llm_model`: LiteLLMへ渡すモデル名。
- `llm_api_base`: 任意。省略時はLiteLLMがモデル名から既定エンドポイントを解決する。
- `llm_api_key`: LLM APIキー。認証不要のローカルエンドポイントでは空文字を明示できる。
- `llm_timeout_seconds`: 任意。1回のLLM呼び出しを待つ秒数。正の整数を指定し、省略時はLiteLLMの既定値を使う。
- `system_prompt_file`: 任意。TOMLファイルからの相対パス。省略時は担当者固有のsystem messageを追加しない。

`id` と `redmine_user_id` はそれぞれ重複できない。指定されたsystem promptファイルは存在し、空でない必要がある。有効なプロフィールが1件もない設定は起動時に拒否する。

今回はAPIキーをTOMLから直接読む。SecretVaultや暗号化された参照は扱わない。将来のSecretVault対応では、TOML解析と実行時プロフィール生成の境界で秘密情報の取得元を差し替える。

## 共有設定

Redmine URL、ステータスID、LinkAce、LangGraph checkpointなど全エージェントで共通の設定は引き続き `.env` と環境変数から読む。

単一エージェント用だった `REDMINE_AI_USER_ID`、`REDMINE_API_KEY`、`LLM_MODEL` は使用しない。エージェント固有設定の旧環境変数へのフォールバックも行わない。

## LLM設定の適用範囲

プロフィールのモデル、API endpoint、API key、タイムアウトは、LiteLLM直接呼び出しとChatLiteLLM経由のLangChain agentの両方へ同じ値を渡す。これにより、計画とtool/skill実行で接続先や待機時間が分かれることを防ぐ。

system promptがある場合は、計画、再計画、LLM step、tool/skill agent、toolやscripted skill内部のLLM処理へ補助system messageとして適用する。共通のStructured Outputs、tool policy、dry-run、承認、Redmine更新規則を優先し、プロフィールのpromptからそれらを変更できないようにする。

## CLIと巡回規則

1件実行ではプロフィールを明示する。

```powershell
uv run taskboard-agent --config agents.toml run-once --agent research-agent
```

`--issue-id` を併用した場合、チケットの現在担当者がプロフィールの `redmine_user_id` と一致しなければ、LLM実行とRedmine更新の前に停止する。

daemonは有効なプロフィールをTOMLの記載順に巡回する。1巡で各プロフィールから最大1件を処理し、全プロフィールが `no_issue` の場合だけ待機する。プロセス内で同時実行は行わない。

LangGraph thread IDは引き続きissue単位とする。担当エージェントが変わっても同じcheckpointから再開し、設定済みAIユーザーの過去コメントは人間の差し戻しとして扱わない。
