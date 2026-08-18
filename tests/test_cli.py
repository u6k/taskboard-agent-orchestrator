from __future__ import annotations

from pathlib import Path

import pytest

from taskboard_agent.cli import build_parser, build_runtime


def test_run_once_parser_accepts_issue_id() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "profiles.toml",
            "run-once",
            "--agent",
            "research",
            "--issue-id",
            "123",
        ]
    )

    assert args.issue_id == 123
    assert args.agent == "research"
    assert args.config == "profiles.toml"


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_run_once_parser_rejects_invalid_issue_id(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["run-once", "--agent", "research", "--issue-id", value]
        )


def test_run_once_parser_requires_agent() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-once"])


def test_run_daemon_parser_uses_default_interval() -> None:
    args = build_parser().parse_args(["run-daemon"])

    assert args.interval_seconds == 60
    assert args.max_iterations is None


def test_run_daemon_parser_accepts_interval_and_max_iterations() -> None:
    args = build_parser().parse_args(
        ["run-daemon", "--interval-seconds", "30", "--max-iterations", "3"]
    )

    assert args.interval_seconds == 30
    assert args.max_iterations == 3


@pytest.mark.parametrize("option", ["--interval-seconds", "--max-iterations"])
@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_run_daemon_parser_rejects_invalid_positive_ints(
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-daemon", option, value])


def test_run_daemon_parser_requires_max_iterations_for_dry_run() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-daemon", "--dry-run"])


def test_run_daemon_parser_accepts_dry_run_with_max_iterations() -> None:
    args = build_parser().parse_args(
        ["run-daemon", "--dry-run", "--max-iterations", "1"]
    )

    assert args.dry_run is True
    assert args.max_iterations == 1


def test_build_runtime_uses_each_profile_connection_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agents_file = tmp_path / "agents.toml"
    agents_file.write_text(
        """
version = 1

[[agents]]
id = "first"
redmine_user_id = 42
redmine_api_key = "redmine-first"
llm_model = "provider/first"
llm_api_base = "https://first.example.test/v1"
llm_api_key = "llm-first"

[[agents]]
id = "second"
redmine_user_id = 43
redmine_api_key = "redmine-second"
llm_model = "provider/second"
llm_api_base = "https://second.example.test/v1"
llm_api_key = "llm-second"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("REDMINE_URL", "https://redmine.example.test")
    monkeypatch.setenv("LINKACE_URL", "https://linkace.example.test")
    monkeypatch.setenv("LINKACE_API_KEY", "linkace-key")
    chat_calls: list[dict[str, object]] = []
    redmine_keys: list[str] = []

    def fake_chat_litellm(**kwargs: object) -> object:
        chat_calls.append(dict(kwargs))
        return object()

    class FakeRedmineClient:
        def __init__(self, _base_url: str, api_key: str) -> None:
            redmine_keys.append(api_key)

    monkeypatch.setattr("taskboard_agent.cli.ChatLiteLLM", fake_chat_litellm)
    monkeypatch.setattr("taskboard_agent.cli.RedmineClient", FakeRedmineClient)

    with build_runtime(dry_run=True, config_path=agents_file) as runtime:
        assert [context.profile.id for context in runtime.agents] == [
            "first",
            "second",
        ]
        direct_llms = [context.task_executor._llm for context in runtime.agents]
        assert [llm.model for llm in direct_llms] == [
            "provider/first",
            "provider/second",
        ]
        assert [llm.api_base for llm in direct_llms] == [
            "https://first.example.test/v1",
            "https://second.example.test/v1",
        ]

    assert redmine_keys == ["redmine-first", "redmine-second"]
    assert chat_calls == [
        {
            "model": "provider/first",
            "api_base": "https://first.example.test/v1",
            "api_key": "llm-first",
        },
        {
            "model": "provider/second",
            "api_base": "https://second.example.test/v1",
            "api_key": "llm-second",
        },
    ]
