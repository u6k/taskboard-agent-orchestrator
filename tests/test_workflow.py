from __future__ import annotations

import logging
from typing import Any

from taskboard_agent.config import AppConfig
from taskboard_agent.skill_runtime import SkillEvent, SkillExecutionResult
from taskboard_agent.workflow import WorkflowError, run_once


CONFIG = AppConfig(
    redmine_url="https://redmine.example.test",
    redmine_api_key="redmine-key",
    redmine_ai_user_id=42,
    redmine_in_progress_status_id=2,
    redmine_review_status_id=10,
    llm_model="test-model",
    linkace_url="https://linkace.example.test",
    linkace_api_key="linkace-key",
    linkace_summarized_list_id=10,
)


class FakeRedmine:
    def __init__(
        self,
        summaries: list[dict[str, Any]],
        issue: dict[str, Any] | None = None,
    ) -> None:
        self.summaries = summaries
        self.issue = issue or {}
        self.updated: list[tuple[int, dict[str, Any]]] = []
        self.requested_assignee: int | None = None

    def find_open_issues_assigned_to(self, assigned_to_id: int) -> list[dict[str, Any]]:
        self.requested_assignee = assigned_to_id
        return self.summaries

    def get_issue(self, issue_id: int) -> dict[str, Any]:
        return self.issue

    def update_issue(
        self,
        issue_id: int,
        *,
        notes: str | None = None,
        assigned_to_id: int | None = None,
        status_id: int | None = None,
        description: str | None = None,
    ) -> None:
        payload = {
            key: value
            for key, value in {
                "notes": notes,
                "assigned_to_id": assigned_to_id,
                "status_id": status_id,
                "description": description,
            }.items()
            if value is not None
        }
        self.updated.append((issue_id, payload))


class FakeTaskExecutor:
    def __init__(self, result: SkillExecutionResult | None = None, *, fail: bool = False) -> None:
        self.result = result or _processed_result()
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        issue: dict[str, Any],
        dry_run: bool = False,
        emit_event: Any | None = None,
    ) -> SkillExecutionResult:
        self.calls.append({"issue": issue, "dry_run": dry_run})
        if self.fail:
            raise RuntimeError("executor failed")
        result = self.result
        if dry_run and result.status == "processed":
            result = SkillExecutionResult(
                status="dry_run",
                events=result.events,
                target_url=result.target_url,
                page_title=result.page_title,
                briefing=result.briefing,
                bookmark_url=result.bookmark_url,
                bookmark_payload=result.bookmark_payload,
                dry_run=True,
            )
        if emit_event is None or not result.events:
            return result
        for event in result.events[:-1]:
            emit_event(event)
        return SkillExecutionResult(
            status=result.status,
            events=result.events[-1:],
            target_url=result.target_url,
            page_title=result.page_title,
            briefing=result.briefing,
            bookmark_url=result.bookmark_url,
            bookmark_payload=result.bookmark_payload,
            dry_run=result.dry_run,
        )


def _issue() -> dict[str, Any]:
    return {"id": 123, "author": {"id": 7}, "subject": "test", "description": "do it"}


def _processed_result() -> SkillExecutionResult:
    return SkillExecutionResult(
        status="processed",
        events=(
            SkillEvent(
                "start",
                "ユーザーから依頼された作業は、https://example.test/article を対象にしたブリーフィング要約とブックマーク登録だと理解しました。URL要約依頼と判断したためです。スキル `web-briefing-bookmark` を使って進めます。\n\n作業を開始します。",
            ),
            SkillEvent("progress", "スキル `web-briefing-bookmark` を実行しました。"),
            SkillEvent(
                "final_review",
                "作業を完了しました。\n\n- ブックマーク: https://linkace.example.test/links/99\n- 要約: 要約本文",
            ),
        ),
        target_url="https://example.test/article",
        page_title="Article title",
        briefing="要約本文",
        bookmark_url="https://linkace.example.test/links/99",
        bookmark_payload={
            "url": "https://example.test/article",
            "title": "Article title",
            "description": "要約本文",
            "list_id": 10,
        },
    )


def test_run_once_no_issue_does_not_execute() -> None:
    redmine = FakeRedmine([])
    executor = FakeTaskExecutor()

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        task_executor=executor,
    )

    assert result.status == "no_issue"
    assert redmine.requested_assignee == 42
    assert executor.calls == []
    assert redmine.updated == []


def test_run_once_applies_skill_events_to_redmine() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        task_executor=FakeTaskExecutor(),
    )

    assert result.status == "processed"
    assert result.issue_id == 123
    assert result.reassigned_to_id == 7
    assert result.target_url == "https://example.test/article"
    assert result.page_title == "Article title"
    assert result.briefing == "要約本文"
    assert result.bookmark_url == "https://linkace.example.test/links/99"
    assert redmine.updated == [
        (
            123,
            {
                "notes": "ユーザーから依頼された作業は、https://example.test/article を対象にしたブリーフィング要約とブックマーク登録だと理解しました。URL要約依頼と判断したためです。スキル `web-briefing-bookmark` を使って進めます。\n\n作業を開始します。",
                "status_id": 2,
            },
        ),
        (123, {"notes": "スキル `web-briefing-bookmark` を実行しました。"}),
        (
            123,
            {
                "notes": "作業を完了しました。\n\n- ブックマーク: https://linkace.example.test/links/99\n- 要約: 要約本文",
                "assigned_to_id": 7,
                "status_id": 10,
            },
        ),
    ]


def test_run_once_needs_user_returns_to_author_for_review() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())
    executor = FakeTaskExecutor(
        SkillExecutionResult(
            status="needs_user",
            events=(SkillEvent("final_review", "対象URLを追記してください。"),),
        )
    )

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        task_executor=executor,
    )

    assert result.status == "needs_user"
    assert result.comments == ("対象URLを追記してください。",)
    assert redmine.updated == [
        (
            123,
            {
                "notes": "対象URLを追記してください。",
                "assigned_to_id": 7,
                "status_id": 10,
            },
        )
    ]


def test_run_once_final_return_returns_to_author_for_review() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())
    executor = FakeTaskExecutor(
        SkillExecutionResult(
            status="missing_tool",
            events=(SkillEvent("final_return", "必要なtoolがありません。"),),
        )
    )

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        task_executor=executor,
    )

    assert result.status == "missing_tool"
    assert redmine.updated == [
        (
            123,
            {
                "notes": "必要なtoolがありません。",
                "assigned_to_id": 7,
                "status_id": 10,
            },
        )
    ]


def test_run_once_can_return_to_review_without_adding_a_comment() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())
    executor = FakeTaskExecutor(
        SkillExecutionResult(
            status="processed",
            events=(SkillEvent("final_review", None),),
        )
    )

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        task_executor=executor,
    )

    assert result.status == "processed"
    assert result.comments == ()
    assert redmine.updated == [
        (123, {"assigned_to_id": 7, "status_id": 10})
    ]


def test_run_once_dry_run_does_not_update_redmine() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        task_executor=FakeTaskExecutor(),
        dry_run=True,
    )

    assert result.status == "dry_run"
    assert result.dry_run is True
    assert result.bookmark_payload == {
        "url": "https://example.test/article",
        "title": "Article title",
        "description": "要約本文",
        "list_id": 10,
    }
    assert redmine.updated == []


def test_run_once_executor_failure_comments_and_returns_to_author() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        task_executor=FakeTaskExecutor(fail=True),
    )

    assert result.status == "agent_failed"
    assert "AIエージェントの実行に失敗" in result.comments[-1]
    assert redmine.updated[-1] == (
        123,
        {
            "notes": result.comments[-1],
            "assigned_to_id": 7,
            "status_id": 10,
        },
    )


def test_run_once_logs_progress_with_trace_id(caplog) -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())

    caplog.set_level(logging.INFO, logger="taskboard_agent.workflow")

    run_once(
        config=CONFIG,
        redmine=redmine,
        task_executor=FakeTaskExecutor(),
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Redmineの未完了チケットを検索します" in message for message in messages)
    assert any("依頼内容を理解し実行方法を決定します" in message for message in messages)
    assert any(record.trace_id == "issue#123" for record in caplog.records)


def test_run_once_missing_author_id_fails_without_update() -> None:
    redmine = FakeRedmine(
        [{"id": 123}],
        issue={"id": 123, "author": {"name": "requester"}, "subject": "test"},
    )

    try:
        run_once(
            config=CONFIG,
            redmine=redmine,
            task_executor=FakeTaskExecutor(),
        )
    except WorkflowError as exc:
        assert "author did not include an integer id" in str(exc)
    else:
        raise AssertionError("WorkflowError was not raised")

    assert redmine.updated == []
