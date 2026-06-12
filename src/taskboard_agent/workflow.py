from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from taskboard_agent.config import AppConfig
from taskboard_agent.llm import CommentGenerationError


class WorkflowError(RuntimeError):
    """Raised when the one-shot workflow cannot complete safely."""


class RedminePort(Protocol):
    def find_open_issues_assigned_to(self, assigned_to_id: int) -> list[dict[str, Any]]:
        ...

    def get_issue(self, issue_id: int) -> dict[str, Any]:
        ...

    def update_description_note_and_reassign(
        self,
        issue_id: int,
        *,
        description: str,
        notes: str,
        assigned_to_id: int,
    ) -> None:
        ...


class DescriptionGeneratorPort(Protocol):
    def generate(self, issue: dict[str, Any]) -> str:
        ...


@dataclass(frozen=True)
class RunResult:
    status: str
    issue_id: int | None = None
    reassigned_to_id: int | None = None
    comment: str | None = None
    description: str | None = None
    dry_run: bool = False


def run_once(
    *,
    config: AppConfig,
    redmine: RedminePort,
    description_generator: DescriptionGeneratorPort,
    dry_run: bool = False,
) -> RunResult:
    summaries = redmine.find_open_issues_assigned_to(config.redmine_ai_user_id)
    if not summaries:
        return RunResult(status="no_issue", dry_run=dry_run)

    issue_id = _require_issue_id(summaries[0])
    issue = redmine.get_issue(issue_id)
    author_id = _require_author_id(issue)

    try:
        description = description_generator.generate(issue)
    except CommentGenerationError:
        raise
    except Exception as exc:
        raise WorkflowError(f"failed to generate updated description: {exc}") from exc

    comment = build_update_comment()

    if dry_run:
        return RunResult(
            status="dry_run",
            issue_id=issue_id,
            reassigned_to_id=author_id,
            comment=comment,
            description=description,
            dry_run=True,
        )

    redmine.update_description_note_and_reassign(
        issue_id,
        description=description,
        notes=comment,
        assigned_to_id=author_id,
    )
    return RunResult(
        status="processed",
        issue_id=issue_id,
        reassigned_to_id=author_id,
        comment=comment,
        description=description,
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


def build_update_comment() -> str:
    return (
        "AIがチケット内容を読み取り、Descriptionを目的・実施すべき内容・完了条件・"
        "課題・元文章の形式に整理しました。\n"
        "作業自体はまだ実行していません。課題欄の不明点を確認し、必要事項を追記したうえで、"
        "再度AI担当にしてください。担当者はチケット作成者へ戻しました。"
    )
