from __future__ import annotations

from typing import Any

import pytest

from taskboard_agent.config import AppConfig
from taskboard_agent.llm import CommentGenerationError
from taskboard_agent.workflow import WorkflowError, run_once


CONFIG = AppConfig(
    redmine_url="https://redmine.example.test",
    redmine_api_key="redmine-key",
    redmine_ai_user_id=42,
    openai_api_key="openai-key",
    openai_model="test-model",
)


class FakeRedmine:
    def __init__(
        self,
        summaries: list[dict[str, Any]],
        issue: dict[str, Any] | None = None,
    ) -> None:
        self.summaries = summaries
        self.issue = issue or {}
        self.updated: list[tuple[int, str, str, int]] = []
        self.requested_assignee: int | None = None

    def find_open_issues_assigned_to(self, assigned_to_id: int) -> list[dict[str, Any]]:
        self.requested_assignee = assigned_to_id
        return self.summaries

    def get_issue(self, issue_id: int) -> dict[str, Any]:
        return self.issue

    def update_description_note_and_reassign(
        self,
        issue_id: int,
        *,
        description: str,
        notes: str,
        assigned_to_id: int,
    ) -> None:
        self.updated.append((issue_id, description, notes, assigned_to_id))


class FakeDescriptionGenerator:
    def __init__(self, description: str = "# 目的\n整理済み") -> None:
        self.description = description
        self.calls: list[dict[str, Any]] = []

    def generate(self, issue: dict[str, Any]) -> str:
        self.calls.append(issue)
        return self.description


class FailingDescriptionGenerator:
    def generate(self, issue: dict[str, Any]) -> str:
        raise CommentGenerationError("LLM failed")


def test_run_once_no_issue_does_not_generate_or_update() -> None:
    redmine = FakeRedmine([])
    generator = FakeDescriptionGenerator()

    result = run_once(config=CONFIG, redmine=redmine, description_generator=generator)

    assert result.status == "no_issue"
    assert redmine.requested_assignee == 42
    assert generator.calls == []
    assert redmine.updated == []


def test_run_once_updates_one_open_ai_issue_and_reassigns_to_author() -> None:
    redmine = FakeRedmine(
        [{"id": 123}],
        issue={"id": 123, "author": {"id": 7}, "subject": "test"},
    )
    generator = FakeDescriptionGenerator("# 目的\n整理済み")

    result = run_once(config=CONFIG, redmine=redmine, description_generator=generator)

    assert result.status == "processed"
    assert result.issue_id == 123
    assert result.reassigned_to_id == 7
    assert result.description == "# 目的\n整理済み"
    assert result.comment is not None
    assert "Descriptionを目的・実施すべき内容・完了条件" in result.comment
    assert "再度AI担当にしてください" in result.comment
    assert redmine.updated == [(123, "# 目的\n整理済み", result.comment, 7)]


def test_run_once_llm_failure_does_not_update_redmine() -> None:
    redmine = FakeRedmine(
        [{"id": 123}],
        issue={"id": 123, "author": {"id": 7}, "subject": "test"},
    )

    with pytest.raises(CommentGenerationError):
        run_once(
            config=CONFIG,
            redmine=redmine,
            description_generator=FailingDescriptionGenerator(),
        )

    assert redmine.updated == []


def test_run_once_missing_author_id_fails_without_update() -> None:
    redmine = FakeRedmine(
        [{"id": 123}],
        issue={"id": 123, "author": {"name": "requester"}, "subject": "test"},
    )
    generator = FakeDescriptionGenerator()

    with pytest.raises(WorkflowError, match="author did not include an integer id"):
        run_once(config=CONFIG, redmine=redmine, description_generator=generator)

    assert generator.calls == []
    assert redmine.updated == []


def test_run_once_dry_run_generates_comment_without_update() -> None:
    redmine = FakeRedmine(
        [{"id": 123}],
        issue={"id": 123, "author": {"id": 7}, "subject": "test"},
    )
    generator = FakeDescriptionGenerator("# 目的\n整理済み")

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        description_generator=generator,
        dry_run=True,
    )

    assert result.status == "dry_run"
    assert result.description == "# 目的\n整理済み"
    assert result.comment is not None
    assert "作業自体はまだ実行していません" in result.comment
    assert result.reassigned_to_id == 7
    assert redmine.updated == []
