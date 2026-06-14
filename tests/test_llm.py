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
        return {"choices": [{"message": {"content": "こんにちは"}}]}

    monkeypatch.setattr("taskboard_agent.llm.litellm.completion", fake_completion)

    response = LiteLLMClient("test-model").complete(
        [{"role": "user", "content": "hello"}]
    )

    assert response.content == "こんにちは"
    assert response.tool_calls == ()


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
