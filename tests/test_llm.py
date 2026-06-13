from __future__ import annotations

import pytest

from taskboard_agent.llm import (
    BRIEFING_PROMPT,
    CommentGenerationError,
    build_briefing_prompt,
    build_issue_prompt,
    build_request_classification_prompt,
    parse_request_classification,
)


def test_build_request_classification_prompt_requires_json_and_includes_issue() -> None:
    prompt = build_request_classification_prompt(
        {
            "subject": "記事を要約する",
            "description": "https://example.test/article をブリーフィング要約してほしい",
        }
    )

    assert "JSON" in prompt
    assert "can_handle" in prompt
    assert "url" in prompt
    assert "reason" in prompt
    assert "記事を要約する" in prompt
    assert "https://example.test/article" in prompt


def test_parse_request_classification_reads_valid_json() -> None:
    classification = parse_request_classification(
        '{"can_handle": true, "url": "https://example.test/article", "reason": "対象"}'
    )

    assert classification.can_handle is True
    assert classification.url == "https://example.test/article"
    assert classification.reason == "対象"


def test_parse_request_classification_rejects_true_without_url() -> None:
    with pytest.raises(CommentGenerationError, match="without url"):
        parse_request_classification(
            '{"can_handle": true, "url": null, "reason": "URLなし"}'
        )


def test_build_briefing_prompt_uses_specified_prompt_and_article_text() -> None:
    prompt = build_briefing_prompt(
        url="https://example.test/article",
        title="Article title",
        text="本文です",
    )

    assert BRIEFING_PROMPT in prompt
    assert "https://example.test/article" in prompt
    assert "Article title" in prompt
    assert "本文です" in prompt


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
