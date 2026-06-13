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
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "LINKACE_URL",
    "LINKACE_API_KEY",
    "LINKACE_SUMMARIZED_LIST_ID",
]


def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_config_reads_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "REDMINE_URL=https://redmine.example.test/",
                "REDMINE_API_KEY=redmine-key",
                "REDMINE_AI_USER_ID=42",
                "REDMINE_IN_PROGRESS_STATUS_ID=2",
                "REDMINE_REVIEW_STATUS_ID=10",
                "OPENAI_API_KEY=openai-key",
                "OPENAI_MODEL=test-model",
                "LINKACE_URL=https://linkace.example.test/",
                "LINKACE_API_KEY=linkace-key",
                "LINKACE_SUMMARIZED_LIST_ID=10",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(env_file)

    assert config.redmine_url == "https://redmine.example.test"
    assert config.redmine_api_key == "redmine-key"
    assert config.redmine_ai_user_id == 42
    assert config.redmine_in_progress_status_id == 2
    assert config.redmine_review_status_id == 10
    assert config.openai_api_key == "openai-key"
    assert config.openai_model == "test-model"
    assert config.linkace_url == "https://linkace.example.test"
    assert config.linkace_api_key == "linkace-key"
    assert config.linkace_summarized_list_id == 10


def test_load_config_real_env_overrides_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "REDMINE_URL=https://from-dotenv.example.test",
                "REDMINE_API_KEY=redmine-key",
                "REDMINE_AI_USER_ID=42",
                "OPENAI_API_KEY=openai-key",
                "OPENAI_MODEL=test-model",
                "LINKACE_URL=https://linkace.example.test",
                "LINKACE_API_KEY=linkace-key",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("REDMINE_URL", "https://from-env.example.test")

    config = load_config(env_file)

    assert config.redmine_url == "https://from-env.example.test"


def test_load_config_requires_integer_ai_user_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "REDMINE_URL=https://redmine.example.test",
                "REDMINE_API_KEY=redmine-key",
                "REDMINE_AI_USER_ID=not-an-int",
                "OPENAI_API_KEY=openai-key",
                "OPENAI_MODEL=test-model",
                "LINKACE_URL=https://linkace.example.test",
                "LINKACE_API_KEY=linkace-key",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="REDMINE_AI_USER_ID must be an integer"):
        load_config(env_file)
