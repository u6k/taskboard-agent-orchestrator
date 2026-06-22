from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from taskboard_agent.config import AppConfig
from taskboard_agent.logging_config import log_trace
from taskboard_agent.skill_runtime import SkillEvent, SkillEventSink, SkillExecutionResult


logger = logging.getLogger(__name__)


class WorkflowError(RuntimeError):
    """Raised when the one-shot workflow cannot complete safely."""


class RedminePort(Protocol):
    def find_open_issues_assigned_to(self, assigned_to_id: int) -> list[dict[str, Any]]:
        ...

    def get_issue(self, issue_id: int) -> dict[str, Any]:
        ...

    def update_issue(
        self,
        issue_id: int,
        *,
        notes: str | None = None,
        assigned_to_id: int | None = None,
        status_id: int | None = None,
        description: str | None = None,
    ) -> None:
        ...


class TaskExecutorPort(Protocol):
    def run(
        self,
        *,
        issue: dict[str, Any],
        dry_run: bool = False,
        emit_event: SkillEventSink | None = None,
    ) -> SkillExecutionResult:
        ...


@dataclass(frozen=True)
class RunResult:
    status: str
    issue_id: int | None = None
    reassigned_to_id: int | None = None
    comments: tuple[str, ...] = ()
    target_url: str | None = None
    page_title: str | None = None
    briefing: str | None = None
    bookmark_url: str | None = None
    bookmark_payload: dict[str, Any] | None = None
    dry_run: bool = False


def run_once(
    *,
    config: AppConfig,
    redmine: RedminePort,
    task_executor: TaskExecutorPort,
    dry_run: bool = False,
    issue_id: int | None = None,
) -> RunResult:
    if issue_id is None:
        with log_trace("run-once"):
            logger.info(
                "Redmineの未完了チケットを検索します assigned_to_id=%s dry_run=%s",
                config.redmine_ai_user_id,
                dry_run,
            )
            summaries = redmine.find_open_issues_assigned_to(config.redmine_ai_user_id)
            if not summaries:
                logger.warning(
                    "処理対象のチケットがありません assigned_to_id=%s status=no_issue",
                    config.redmine_ai_user_id,
                )
                return RunResult(status="no_issue", dry_run=dry_run)
        issue_id = _require_issue_id(summaries[0])
    elif issue_id <= 0:
        raise WorkflowError("issue_id must be a positive integer")
    else:
        with log_trace(f"issue#{issue_id}"):
            logger.info(
                "CLIで指定されたRedmineチケットを処理します issue_id=%s dry_run=%s",
                issue_id,
                dry_run,
            )

    with log_trace(f"issue#{issue_id}"):
        logger.info("Redmineチケットを取得します issue_id=%s", issue_id)
        issue = redmine.get_issue(issue_id)
        author_id = _require_author_id(issue)
        logger.info("Redmineチケットを取得しました issue_id=%s author_id=%s", issue_id, author_id)
        comments: list[str] = []

        logger.info("依頼内容を理解し実行方法を決定します issue_id=%s", issue_id)
        try:
            def emit_event(event: SkillEvent) -> None:
                _apply_skill_event(
                    redmine=redmine,
                    issue_id=issue_id,
                    author_id=author_id,
                    config=config,
                    event=event,
                    comments=comments,
                    dry_run=dry_run,
                )

            execution = task_executor.run(
                issue=issue,
                dry_run=dry_run,
                emit_event=emit_event,
            )
        except Exception as exc:
            logger.warning("依頼内容の実行中に例外が発生しました issue_id=%s", issue_id, exc_info=True)
            comment = f"AIエージェントの実行に失敗したため、担当者を戻します。\n理由: {exc}"
            comments.append(comment)
            if not dry_run:
                logger.info(
                    "Redmineチケットへ実行失敗コメントを追加し担当者を戻します issue_id=%s assigned_to_id=%s",
                    issue_id,
                    author_id,
                )
                redmine.update_issue(
                    issue_id,
                    notes=comment,
                    assigned_to_id=author_id,
                    status_id=config.redmine_review_status_id,
                )
            return RunResult(
                status="agent_failed",
                issue_id=issue_id,
                reassigned_to_id=author_id,
                comments=tuple(comments),
                dry_run=dry_run,
            )

        _apply_skill_events(
            redmine=redmine,
            issue_id=issue_id,
            author_id=author_id,
            config=config,
            execution=execution,
            comments=comments,
            dry_run=dry_run,
        )

        if execution.status == "dry_run":
            logger.info("dry-runのため外部サービスを更新せず終了します status=dry_run")
            return RunResult(
                status="dry_run",
                issue_id=issue_id,
                reassigned_to_id=author_id,
                comments=tuple(comments),
                target_url=execution.target_url,
                page_title=execution.page_title,
                briefing=execution.briefing,
                bookmark_payload=execution.bookmark_payload,
                dry_run=True,
            )

        logger.info(
            "ワークフローを完了しました status=%s target_url=%s bookmark_url=%s",
            execution.status,
            execution.target_url,
            execution.bookmark_url,
        )
        return RunResult(
            status=execution.status,
            issue_id=issue_id,
            reassigned_to_id=author_id,
            comments=tuple(comments),
            target_url=execution.target_url,
            page_title=execution.page_title,
            briefing=execution.briefing,
            bookmark_url=execution.bookmark_url,
            bookmark_payload=execution.bookmark_payload,
            dry_run=dry_run,
        )


def _apply_skill_events(
    *,
    redmine: RedminePort,
    issue_id: int,
    author_id: int,
    config: AppConfig,
    execution: SkillExecutionResult,
    comments: list[str],
    dry_run: bool,
) -> None:
    for event in execution.events:
        _apply_skill_event(
            redmine=redmine,
            issue_id=issue_id,
            author_id=author_id,
            config=config,
            event=event,
            comments=comments,
            dry_run=dry_run,
        )


def _apply_skill_event(
    *,
    redmine: RedminePort,
    issue_id: int,
    author_id: int,
    config: AppConfig,
    event: SkillEvent,
    comments: list[str],
    dry_run: bool,
) -> None:
    if event.notes is not None:
        comments.append(event.notes)
    if dry_run:
        return
    if event.kind == "start":
        redmine.update_issue(
            issue_id,
            notes=event.notes,
            status_id=config.redmine_in_progress_status_id,
        )
    elif event.kind == "progress":
        if event.notes is not None:
            redmine.update_issue(issue_id, notes=event.notes)
    elif event.kind == "final_review":
        redmine.update_issue(
            issue_id,
            notes=event.notes,
            assigned_to_id=author_id,
            status_id=config.redmine_review_status_id,
        )
    elif event.kind == "final_return":
        redmine.update_issue(
            issue_id,
            notes=event.notes,
            assigned_to_id=author_id,
            status_id=config.redmine_review_status_id,
        )


def _require_issue_id(issue_summary: dict[str, Any]) -> int:
    issue_id = issue_summary.get("id")
    if not isinstance(issue_id, int):
        raise WorkflowError("Redmine issue summary did not include an integer id")
    return issue_id


def _require_author_id(issue: dict[str, Any]) -> int:
    author = issue.get("author")
    if not isinstance(author, dict):
        raise WorkflowError("Redmine issue did not include an author")
    author_id = author.get("id")
    if not isinstance(author_id, int):
        raise WorkflowError("Redmine issue author did not include an integer id")
    return author_id
