from __future__ import annotations

from taskboard_agent.llm import build_issue_prompt


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

