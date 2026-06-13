from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from taskboard_agent.config import AppConfig
from taskboard_agent.linkace import BookmarkResult, ExistingBookmark
from taskboard_agent.llm import CommentGenerationError, RequestClassification
from taskboard_agent.page import PageContent


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


class RequestClassifierPort(Protocol):
    def classify(self, issue: dict[str, Any]) -> RequestClassification:
        ...


class PageFetcherPort(Protocol):
    def fetch(self, url: str) -> PageContent:
        ...


class BriefingSummarizerPort(Protocol):
    def summarize(self, *, url: str, title: str, text: str) -> str:
        ...


class BookmarkPort(Protocol):
    def check_auth(self) -> None:
        ...

    def find_link(self, url: str) -> ExistingBookmark | None:
        ...

    def add_link(
        self,
        *,
        url: str,
        title: str,
        description: str,
        list_id: int,
    ) -> BookmarkResult:
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
    request_classifier: RequestClassifierPort,
    page_fetcher: PageFetcherPort,
    briefing_summarizer: BriefingSummarizerPort,
    bookmark_client: BookmarkPort,
    dry_run: bool = False,
) -> RunResult:
    summaries = redmine.find_open_issues_assigned_to(config.redmine_ai_user_id)
    if not summaries:
        return RunResult(status="no_issue", dry_run=dry_run)

    issue_id = _require_issue_id(summaries[0])
    issue = redmine.get_issue(issue_id)
    author_id = _require_author_id(issue)
    comments: list[str] = []

    try:
        classification = request_classifier.classify(issue)
    except CommentGenerationError as exc:
        comment = f"AI判定に失敗したため、担当者を戻します。\n理由: {exc}"
        comments.append(comment)
        if not dry_run:
            redmine.update_issue(issue_id, notes=comment, assigned_to_id=author_id)
        return RunResult(
            status="classification_failed",
            issue_id=issue_id,
            reassigned_to_id=author_id,
            comments=tuple(comments),
            dry_run=dry_run,
        )
    except Exception as exc:
        raise WorkflowError(f"failed to classify issue request: {exc}") from exc

    if not classification.can_handle or classification.url is None:
        comment = build_unhandled_comment(classification.reason)
        comments.append(comment)
        if not dry_run:
            redmine.update_issue(issue_id, notes=comment, assigned_to_id=author_id)
        return RunResult(
            status="unsupported",
            issue_id=issue_id,
            reassigned_to_id=author_id,
            comments=tuple(comments),
            dry_run=dry_run,
        )

    if not dry_run:
        try:
            bookmark_client.check_auth()
        except Exception as exc:
            comment = f"LinkAce認証に失敗したため、担当者を戻します。\n理由: {exc}"
            comments.append(comment)
            redmine.update_issue(issue_id, notes=comment, assigned_to_id=author_id)
            return RunResult(
                status="linkace_auth_failed",
                issue_id=issue_id,
                reassigned_to_id=author_id,
                comments=tuple(comments),
                target_url=classification.url,
                dry_run=dry_run,
            )

    if not dry_run:
        existing = bookmark_client.find_link(classification.url)
        if existing is not None and not existing.has_list(1):
            comment = build_already_bookmarked_comment(existing)
            comments.append(comment)
            redmine.update_issue(
                issue_id,
                notes=comment,
                assigned_to_id=author_id,
                status_id=config.redmine_review_status_id,
            )
            return RunResult(
                status="already_bookmarked",
                issue_id=issue_id,
                reassigned_to_id=author_id,
                comments=tuple(comments),
                target_url=classification.url,
                bookmark_url=existing.web_url,
                dry_run=dry_run,
            )

    start_comment = "作業を開始します。"
    comments.append(start_comment)
    if not dry_run:
        redmine.update_issue(
            issue_id,
            notes=start_comment,
            status_id=config.redmine_in_progress_status_id,
        )

    try:
        page = page_fetcher.fetch(classification.url)
    except Exception as exc:
        comment = f"ページ本文を取得できませんでした。\n理由: {exc}"
        comments.append(comment)
        if not dry_run:
            redmine.update_issue(issue_id, notes=comment, assigned_to_id=author_id)
        return RunResult(
            status="page_fetch_failed",
            issue_id=issue_id,
            reassigned_to_id=author_id,
            comments=tuple(comments),
            target_url=classification.url,
            dry_run=dry_run,
        )

    page_comment = "ページ本文を取得しました。"
    comments.append(page_comment)
    if not dry_run:
        redmine.update_issue(issue_id, notes=page_comment)

        if page.url != classification.url:
            existing = bookmark_client.find_link(page.url)
            if existing is not None and not existing.has_list(1):
                comment = build_already_bookmarked_comment(existing)
                comments.append(comment)
                redmine.update_issue(
                    issue_id,
                    notes=comment,
                    assigned_to_id=author_id,
                    status_id=config.redmine_review_status_id,
                )
                return RunResult(
                    status="already_bookmarked",
                    issue_id=issue_id,
                    reassigned_to_id=author_id,
                    comments=tuple(comments),
                    target_url=page.url,
                    page_title=page.title,
                    bookmark_url=existing.web_url,
                    dry_run=dry_run,
                )

    try:
        briefing = briefing_summarizer.summarize(
            url=page.url,
            title=page.title,
            text=page.text,
        )
    except CommentGenerationError as exc:
        comment = f"ブリーフィング要約を生成できませんでした。\n理由: {exc}"
        comments.append(comment)
        if not dry_run:
            redmine.update_issue(issue_id, notes=comment, assigned_to_id=author_id)
        return RunResult(
            status="briefing_failed",
            issue_id=issue_id,
            reassigned_to_id=author_id,
            comments=tuple(comments),
            target_url=page.url,
            page_title=page.title,
            dry_run=dry_run,
        )

    briefing_comment = build_briefing_comment(briefing)
    comments.append(briefing_comment)
    bookmark_payload = build_bookmark_payload(
        url=page.url,
        title=page.title,
        description=briefing,
        list_id=config.linkace_summarized_list_id,
    )
    if not dry_run:
        redmine.update_issue(issue_id, notes=briefing_comment)
    else:
        return RunResult(
            status="dry_run",
            issue_id=issue_id,
            reassigned_to_id=author_id,
            comments=tuple(comments),
            target_url=page.url,
            page_title=page.title,
            briefing=briefing,
            bookmark_payload=bookmark_payload,
            dry_run=True,
        )

    try:
        bookmark = bookmark_client.add_link(**bookmark_payload)
    except Exception as exc:
        comment = f"ブックマークを登録できませんでした。\n理由: {exc}"
        comments.append(comment)
        if not dry_run:
            redmine.update_issue(issue_id, notes=comment, assigned_to_id=author_id)
        return RunResult(
            status="bookmark_failed",
            issue_id=issue_id,
            reassigned_to_id=author_id,
            comments=tuple(comments),
            target_url=page.url,
            page_title=page.title,
            briefing=briefing,
            dry_run=dry_run,
        )

    bookmark_comment = build_bookmark_comment(bookmark)
    comments.append(bookmark_comment)
    if not dry_run:
        redmine.update_issue(
            issue_id,
            notes=bookmark_comment,
            assigned_to_id=author_id,
            status_id=config.redmine_review_status_id,
        )

    return RunResult(
        status="dry_run" if dry_run else "processed",
        issue_id=issue_id,
        reassigned_to_id=author_id,
        comments=tuple(comments),
        target_url=page.url,
        page_title=page.title,
        briefing=briefing,
        bookmark_url=bookmark.url,
        bookmark_payload=bookmark_payload,
        dry_run=dry_run,
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


def build_unhandled_comment(reason: str) -> str:
    if reason:
        return (
            "AIエージェントでは作業できないタイプの依頼なので、担当者を戻します。\n"
            f"判定理由: {reason}"
        )
    return "AIエージェントでは作業できないタイプの依頼なので、担当者を戻します。"


def build_briefing_comment(briefing: str) -> str:
    return f"以下のようにブリーフィング要約を生成しました。\n\n{briefing}"


def build_bookmark_comment(bookmark: BookmarkResult) -> str:
    if bookmark.action == "already_exists":
        if bookmark.url:
            return f"ブックマークが登録済みです。\n{bookmark.url}"
        return "ブックマークが登録済みです。"
    action_text = "更新" if bookmark.action == "updated" else "登録"
    if bookmark.url:
        return f"ブックマークを{action_text}しました。\n{bookmark.url}"
    return f"ブックマークを{action_text}しました。"


def build_already_bookmarked_comment(bookmark: ExistingBookmark) -> str:
    return f"ブックマークが登録済みです。\n{bookmark.web_url}"


def build_bookmark_payload(
    *,
    url: str,
    title: str,
    description: str,
    list_id: int,
) -> dict[str, Any]:
    return {
        "url": url,
        "title": title,
        "description": description,
        "list_id": list_id,
    }
