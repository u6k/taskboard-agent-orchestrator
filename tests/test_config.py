from __future__ import annotations

from pathlib import Path

import pytest

from taskboard_agent.config import ConfigError, load_config


ENV_KEYS = [
    "REDMINE_URL",
    "REDMINE_API_KEY",
    "REDMINE_AI_USER_ID",
    "REDMINE_IN_PROGRESS_STATUS_ID",
    "REDMINE_REVIEW_STATUS_ID",
    "LLM_MODEL",
    "LINKACE_URL",
    "LINKACE_API_KEY",
    "LINKACE_SUMMARIZED_LIST_ID",
    "LANGGRAPH_CHECKPOINT_DB_PATH",
]


def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _write_shared_env(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "REDMINE_URL=https://redmine.example.test/",
                "REDMINE_IN_PROGRESS_STATUS_ID=2",
                "REDMINE_REVIEW_STATUS_ID=10",
                "LINKACE_URL=https://linkace.example.test/",
                "LINKACE_API_KEY=linkace-key",
                "LINKACE_SUMMARIZED_LIST_ID=10",
                "LANGGRAPH_CHECKPOINT_DB_PATH=var/test-checkpoints.sqlite3",
            ]
        ),
        encoding="utf-8",
    )


def test_load_config_reads_multiple_agent_profiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    _write_shared_env(env_file)
    prompt = tmp_path / "prompts" / "research.md"
    prompt.parent.mkdir()
    prompt.write_text("根拠を明示してください。", encoding="utf-8")
    agents_file = tmp_path / "agents.toml"
    agents_file.write_text(
        """
version = 1

[[agents]]
id = "research"
redmine_user_id = 42
redmine_api_key = "redmine-research-key"
llm_model = "openai/test-model"
llm_api_base = "https://llm.example.test/v1"
llm_api_key = "llm-research-key"
system_prompt_file = "prompts/research.md"

[[agents]]
id = "local"
redmine_user_id = 43
redmine_api_key = "redmine-local-key"
llm_model = "lm_studio/local-model"
llm_api_key = ""
""".strip(),
        encoding="utf-8",
    )

    config = load_config(env_file, agents_file)

    assert config.redmine_url == "https://redmine.example.test"
    assert config.linkace_url == "https://linkace.example.test"
    assert config.langgraph_checkpoint_db_path == Path(
        "var/test-checkpoints.sqlite3"
    )
    assert [agent.id for agent in config.agents] == ["research", "local"]
    research, local = config.agents
    assert research.redmine_user_id == 42
    assert research.redmine_api_key == "redmine-research-key"
    assert research.llm_model == "openai/test-model"
    assert research.llm_api_base == "https://llm.example.test/v1"
    assert research.llm_api_key == "llm-research-key"
    assert research.system_prompt == "根拠を明示してください。"
    assert research.system_prompt_file == prompt
    assert local.llm_api_base is None
    assert local.llm_api_key == ""
    assert local.system_prompt is None
    assert local.system_prompt_file is None


def test_load_config_real_env_overrides_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    _write_shared_env(env_file)
    agents_file = _write_minimal_agents(tmp_path)
    monkeypatch.setenv("REDMINE_URL", "https://from-env.example.test")

    config = load_config(env_file, agents_file)

    assert config.redmine_url == "https://from-env.example.test"


@pytest.mark.parametrize(
    ("extra_profile", "message"),
    [
        (
            """
[[agents]]
id = "primary"
redmine_user_id = 43
redmine_api_key = "other"
llm_model = "other"
llm_api_key = "other"
""",
            "duplicate agent id",
        ),
        (
            """
[[agents]]
id = "secondary"
redmine_user_id = 42
redmine_api_key = "other"
llm_model = "other"
llm_api_key = "other"
""",
            "duplicate agent redmine_user_id",
        ),
    ],
)
def test_load_config_rejects_duplicate_agent_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_profile: str,
    message: str,
) -> None:
    clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    _write_shared_env(env_file)
    agents_file = _write_minimal_agents(tmp_path, suffix=extra_profile)

    with pytest.raises(ConfigError, match=message):
        load_config(env_file, agents_file)


def test_load_config_rejects_missing_prompt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    _write_shared_env(env_file)
    agents_file = _write_minimal_agents(
        tmp_path,
        profile_extra='system_prompt_file = "missing.md"',
    )

    with pytest.raises(ConfigError, match="system_prompt_file not found"):
        load_config(env_file, agents_file)


def test_load_config_rejects_all_agents_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    _write_shared_env(env_file)
    agents_file = _write_minimal_agents(tmp_path, profile_extra="enabled = false")

    with pytest.raises(ConfigError, match="enable at least one agent"):
        load_config(env_file, agents_file)


def test_load_config_does_not_fall_back_to_legacy_agent_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    _write_shared_env(env_file)
    monkeypatch.setenv("REDMINE_AI_USER_ID", "42")
    monkeypatch.setenv("REDMINE_API_KEY", "legacy-redmine-key")
    monkeypatch.setenv("LLM_MODEL", "legacy-model")

    with pytest.raises(ConfigError, match="agent config file not found"):
        load_config(env_file, tmp_path / "missing.toml")


def _write_minimal_agents(
    tmp_path: Path,
    *,
    profile_extra: str = "",
    suffix: str = "",
) -> Path:
    agents_file = tmp_path / "agents.toml"
    agents_file.write_text(
        f"""
version = 1

[[agents]]
id = "primary"
redmine_user_id = 42
redmine_api_key = "redmine-key"
llm_model = "test-model"
llm_api_key = "llm-key"
{profile_extra}
{suffix}
""".strip(),
        encoding="utf-8",
    )
    return agents_file
