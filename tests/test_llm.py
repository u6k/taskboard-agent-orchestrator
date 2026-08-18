from __future__ import annotations

import logging

import pytest

from taskboard_agent.llm import (
    LiteLLMClient,
    build_issue_prompt,
)


def test_build_issue_prompt_requires_updated_description_format() -> None:
    prompt = build_issue_prompt(
        {
            "id": 123,
            "subject": "資料を入手する",
            "description": "元の依頼文です。",
            "author": {"id": 7, "name": "requester"},
        }
    )

    assert "# 目的" in prompt
    assert "# 実施すべき内容" in prompt
    assert "# 完了条件" in prompt
    assert "# 課題" in prompt
    assert "# ユーザーが記載した元文章" in prompt
    assert "ユーザーが記載した元Description:" in prompt
    assert "元の依頼文です。" in prompt


def test_litellm_client_reads_text_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: object) -> dict[str, object]:
        assert kwargs["model"] == "test-model"
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
        assert "timeout" not in kwargs
        return {"choices": [{"message": {"content": "こんにちは"}}]}

    monkeypatch.setattr("taskboard_agent.llm.litellm.completion", fake_completion)

    response = LiteLLMClient("test-model").complete(
        [{"role": "user", "content": "hello"}]
    )

    assert response.content == "こんにちは"
    assert response.tool_calls == ()


def test_litellm_client_applies_profile_connection_and_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_messages = [
        {"role": "system", "content": "共通規則"},
        {"role": "user", "content": "hello"},
    ]

    def fake_completion(**kwargs: object) -> dict[str, object]:
        assert kwargs["model"] == "provider/test-model"
        assert kwargs["base_url"] == "https://llm.example.test/v1"
        assert kwargs["api_key"] == "profile-key"
        assert kwargs["timeout"] == 1200
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        assert messages[0] == {"role": "system", "content": "共通規則"}
        assert messages[1]["role"] == "system"
        assert "担当エージェント固有" in messages[1]["content"]
        assert "根拠を明示する" in messages[1]["content"]
        assert messages[2] == {"role": "user", "content": "hello"}
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("taskboard_agent.llm.litellm.completion", fake_completion)

    LiteLLMClient(
        "provider/test-model",
        api_base="https://llm.example.test/v1",
        api_key="profile-key",
        timeout_seconds=1200,
        system_prompt="根拠を明示する",
    ).complete(original_messages)

    assert original_messages == [
        {"role": "system", "content": "共通規則"},
        {"role": "user", "content": "hello"},
    ]


def test_litellm_client_does_not_add_empty_profile_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_completion(**kwargs: object) -> dict[str, object]:
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("taskboard_agent.llm.litellm.completion", fake_completion)

    LiteLLMClient("test-model", system_prompt=None).complete(
        [{"role": "user", "content": "hello"}]
    )


def test_litellm_client_logs_prompt_and_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_completion(**kwargs: object) -> dict[str, object]:
        return {"choices": [{"message": {"content": "こんにちは"}}]}

    monkeypatch.setattr("taskboard_agent.llm.litellm.completion", fake_completion)
    caplog.set_level(logging.DEBUG, logger="taskboard_agent.llm")

    LiteLLMClient("test-model").complete(
        [{"role": "user", "content": "hello"}]
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("LLM入力プロンプト" in message and "hello" in message for message in messages)
    assert any("LLM出力内容" in message and "こんにちは" in message for message in messages)


def test_litellm_client_reads_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: object) -> dict[str, object]:
        assert kwargs["tools"] == [{"type": "function", "function": {"name": "echo"}}]
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"text": "hello"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr("taskboard_agent.llm.litellm.completion", fake_completion)

    response = LiteLLMClient("test-model").complete(
        [{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "echo"}}],
    )

    assert response.content == ""
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].name == "echo"
    assert response.tool_calls[0].arguments == '{"text": "hello"}'


def test_litellm_client_passes_strict_json_schema_and_enables_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
                "additionalProperties": False,
            },
        },
    }

    def fake_completion(**kwargs: object) -> dict[str, object]:
        assert kwargs["response_format"] == response_format
        assert kwargs["enable_json_schema_validation"] is True
        return {"choices": [{"message": {"content": '{"status":"ok"}'}}]}

    monkeypatch.setattr("taskboard_agent.llm.litellm.completion", fake_completion)

    response = LiteLLMClient("test-model").complete(
        [{"role": "user", "content": "hello"}],
        response_format=response_format,
    )

    assert response.content == '{"status":"ok"}'
