from __future__ import annotations

import logging
from typing import Any

from taskboard_agent.config import AppConfig
from taskboard_agent.linkace import BookmarkResult, ExistingBookmark
from taskboard_agent.llm import CommentGenerationError, RequestClassification
from taskboard_agent.page import PageContent
from taskboard_agent.workflow import WorkflowError, run_once


CONFIG = AppConfig(
    redmine_url="https://redmine.example.test",
    redmine_api_key="redmine-key",
    redmine_ai_user_id=42,
    redmine_in_progress_status_id=2,
    redmine_review_status_id=10,
    openai_api_key="openai-key",
    openai_model="test-model",
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


class FakeClassifier:
    def __init__(
        self,
        classification: RequestClassification = RequestClassification(
            can_handle=True,
            url="https://example.test/article",
            reason="ブリーフィング要約依頼です",
        ),
        *,
        fail: bool = False,
    ) -> None:
        self.classification = classification
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def classify(self, issue: dict[str, Any]) -> RequestClassification:
        self.calls.append(issue)
        if self.fail:
            raise CommentGenerationError("bad JSON")
        return self.classification


class FakePageFetcher:
    def __init__(
        self,
        *,
        fail: bool = False,
        page_url: str = "https://example.test/article",
    ) -> None:
        self.fail = fail
        self.page_url = page_url
        self.calls: list[str] = []

    def fetch(self, url: str) -> PageContent:
        self.calls.append(url)
        if self.fail:
            raise RuntimeError("HTTP 403")
        return PageContent(url=self.page_url, title="Article title", text="Article body")


class FakeSummarizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, str]] = []

    def summarize(self, *, url: str, title: str, text: str) -> str:
        self.calls.append({"url": url, "title": title, "text": text})
        if self.fail:
            raise CommentGenerationError("LLM failed")
        return "# エグゼクティブサマリー\n要約本文"


class FakeBookmarkClient:
    def __init__(
        self,
        *,
        fail: bool = False,
        auth_fail: bool = False,
        action: str = "created",
        existing_by_url: dict[str, ExistingBookmark] | None = None,
    ) -> None:
        self.fail = fail
        self.auth_fail = auth_fail
        self.action = action
        self.existing_by_url = existing_by_url or {}
        self.auth_checks = 0
        self.find_calls: list[str] = []
        self.calls: list[dict[str, Any]] = []

    def check_auth(self) -> None:
        self.auth_checks += 1
        if self.auth_fail:
            raise RuntimeError("HTTP 401 Unauthenticated")

    def find_link(self, url: str) -> ExistingBookmark | None:
        self.find_calls.append(url)
        return self.existing_by_url.get(url)

    def add_link(
        self,
        *,
        url: str,
        title: str,
        description: str,
        list_id: int,
    ) -> BookmarkResult:
        self.calls.append(
            {
                "url": url,
                "title": title,
                "description": description,
                "list_id": list_id,
            }
        )
        if self.fail:
            raise RuntimeError("LinkAce failed")
        return BookmarkResult(
            id=99,
            url="https://linkace.example.test/links/99",
            action=self.action,
        )


def _issue() -> dict[str, Any]:
    return {"id": 123, "author": {"id": 7}, "subject": "test"}


def test_run_once_no_issue_does_not_classify_or_update() -> None:
    redmine = FakeRedmine([])
    classifier = FakeClassifier()

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=classifier,
        page_fetcher=FakePageFetcher(),
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=FakeBookmarkClient(),
    )

    assert result.status == "no_issue"
    assert redmine.requested_assignee == 42
    assert classifier.calls == []
    assert redmine.updated == []


def test_run_once_returns_unsupported_request_to_author() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())
    classifier = FakeClassifier(
        RequestClassification(can_handle=False, url=None, reason="URLがありません")
    )
    page_fetcher = FakePageFetcher()
    bookmark_client = FakeBookmarkClient()

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=classifier,
        page_fetcher=page_fetcher,
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=bookmark_client,
    )

    assert result.status == "unsupported"
    assert result.reassigned_to_id == 7
    assert "作業できないタイプ" in result.comments[0]
    assert "URLがありません" in result.comments[0]
    assert redmine.updated == [
        (
            123,
            {
                "notes": result.comments[0],
                "assigned_to_id": 7,
            },
        )
    ]
    assert page_fetcher.calls == []
    assert bookmark_client.calls == []


def test_run_once_linkace_auth_failure_returns_before_starting_work() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())
    page_fetcher = FakePageFetcher()
    bookmark_client = FakeBookmarkClient(auth_fail=True)

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=page_fetcher,
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=bookmark_client,
    )

    assert result.status == "linkace_auth_failed"
    assert result.reassigned_to_id == 7
    assert "LinkAce認証に失敗" in result.comments[-1]
    assert redmine.updated == [
        (
            123,
            {
                "notes": result.comments[-1],
                "assigned_to_id": 7,
            },
        )
    ]
    assert page_fetcher.calls == []
    assert bookmark_client.calls == []


def test_run_once_returns_registered_bookmark_before_fetching_when_existing_is_not_source_list() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())
    page_fetcher = FakePageFetcher()
    summarizer = FakeSummarizer()
    bookmark_client = FakeBookmarkClient(
        existing_by_url={
            "https://example.test/article": ExistingBookmark(
                id=44,
                url="https://example.test/article",
                web_url="https://linkace.example.test/links/44",
                list_ids=(10,),
            )
        }
    )

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=page_fetcher,
        briefing_summarizer=summarizer,
        bookmark_client=bookmark_client,
    )

    assert result.status == "already_bookmarked"
    assert result.bookmark_url == "https://linkace.example.test/links/44"
    assert result.comments == (
        "ブックマークが登録済みです。\nhttps://linkace.example.test/links/44",
    )
    assert redmine.updated == [
        (
            123,
            {
                "notes": result.comments[-1],
                "assigned_to_id": 7,
                "status_id": 10,
            },
        )
    ]
    assert page_fetcher.calls == []
    assert summarizer.calls == []
    assert bookmark_client.calls == []


def test_run_once_returns_registered_bookmark_before_summarizing_redirected_url() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())
    summarizer = FakeSummarizer()
    bookmark_client = FakeBookmarkClient(
        existing_by_url={
            "https://example.test/final": ExistingBookmark(
                id=45,
                url="https://example.test/final",
                web_url="https://linkace.example.test/links/45",
                list_ids=(10,),
            )
        }
    )

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=FakePageFetcher(page_url="https://example.test/final"),
        briefing_summarizer=summarizer,
        bookmark_client=bookmark_client,
    )

    assert result.status == "already_bookmarked"
    assert result.bookmark_url == "https://linkace.example.test/links/45"
    assert result.comments[-1] == "ブックマークが登録済みです。\nhttps://linkace.example.test/links/45"
    assert redmine.updated == [
        (123, {"notes": "作業を開始します。", "status_id": 2}),
        (123, {"notes": "ページ本文を取得しました。"}),
        (
            123,
            {
                "notes": result.comments[-1],
                "assigned_to_id": 7,
                "status_id": 10,
            },
        ),
    ]
    assert summarizer.calls == []
    assert bookmark_client.calls == []


def test_run_once_processes_briefing_request_and_returns_for_review() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())
    bookmark_client = FakeBookmarkClient()

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=FakePageFetcher(),
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=bookmark_client,
    )

    assert result.status == "processed"
    assert result.issue_id == 123
    assert result.reassigned_to_id == 7
    assert result.target_url == "https://example.test/article"
    assert result.page_title == "Article title"
    assert result.briefing == "# エグゼクティブサマリー\n要約本文"
    assert result.bookmark_url == "https://linkace.example.test/links/99"
    assert bookmark_client.auth_checks == 1
    assert bookmark_client.calls == [
        {
            "url": "https://example.test/article",
            "title": "Article title",
            "description": "# エグゼクティブサマリー\n要約本文",
            "list_id": 10,
        }
    ]
    assert redmine.updated == [
        (123, {"notes": "作業を開始します。", "status_id": 2}),
        (123, {"notes": "ページ本文を取得しました。"}),
        (123, {"notes": result.comments[2]}),
        (
            123,
            {
                "notes": "ブックマークを登録しました。\nhttps://linkace.example.test/links/99",
                "assigned_to_id": 7,
                "status_id": 10,
            },
        ),
    ]
    assert result.comments[2].startswith("以下のようにブリーフィング要約を生成しました。")


def test_run_once_logs_progress_with_trace_id(caplog) -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())

    caplog.set_level(logging.INFO, logger="taskboard_agent.workflow")

    run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=FakePageFetcher(),
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=FakeBookmarkClient(),
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Redmineの未完了チケットを検索します" in message for message in messages)
    assert any("ページ本文を取得しました" in message for message in messages)
    assert any(
        "LinkAceへブックマークを登録しました" in message
        and "bookmark_id=99" in message
        and "bookmark_url=https://linkace.example.test/links/99" in message
        for message in messages
    )
    assert any(record.trace_id == "issue#123" for record in caplog.records)


def test_run_once_logs_caught_exception_with_stack_trace(caplog) -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())

    caplog.set_level(logging.WARNING, logger="taskboard_agent.workflow")

    run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=FakePageFetcher(fail=True),
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=FakeBookmarkClient(),
    )

    warning_records = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "ページ本文の取得に失敗しました" in record.getMessage()
    ]
    assert len(warning_records) == 1
    assert warning_records[0].trace_id == "issue#123"
    assert warning_records[0].exc_info is not None


def test_run_once_logs_unsupported_request_as_warning(caplog) -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())
    classifier = FakeClassifier(
        RequestClassification(can_handle=False, url=None, reason="URLがありません")
    )

    caplog.set_level(logging.WARNING, logger="taskboard_agent.workflow")

    run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=classifier,
        page_fetcher=FakePageFetcher(),
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=FakeBookmarkClient(),
    )

    assert any(
        record.levelno == logging.WARNING
        and record.trace_id == "issue#123"
        and "処理対象外の依頼です" in record.getMessage()
        for record in caplog.records
    )


def test_run_once_reports_updated_bookmark_when_linkace_updates_existing_link() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=FakePageFetcher(),
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=FakeBookmarkClient(action="updated"),
    )

    assert result.status == "processed"
    assert result.comments[-1] == "ブックマークを更新しました。\nhttps://linkace.example.test/links/99"
    assert redmine.updated[-1] == (
        123,
        {
            "notes": result.comments[-1],
            "assigned_to_id": 7,
            "status_id": 10,
        },
    )


def test_run_once_page_failure_comments_and_returns_to_author_only() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=FakePageFetcher(fail=True),
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=FakeBookmarkClient(),
    )

    assert result.status == "page_fetch_failed"
    assert "ページ本文を取得できませんでした" in result.comments[-1]
    assert redmine.updated[-1] == (
        123,
        {"notes": result.comments[-1], "assigned_to_id": 7},
    )
    assert "status_id" not in redmine.updated[-1][1]


def test_run_once_briefing_failure_comments_and_returns_to_author_only() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=FakePageFetcher(),
        briefing_summarizer=FakeSummarizer(fail=True),
        bookmark_client=FakeBookmarkClient(),
    )

    assert result.status == "briefing_failed"
    assert "ブリーフィング要約を生成できませんでした" in result.comments[-1]
    assert redmine.updated[-1] == (
        123,
        {"notes": result.comments[-1], "assigned_to_id": 7},
    )
    assert "status_id" not in redmine.updated[-1][1]


def test_run_once_bookmark_failure_comments_and_returns_to_author_only() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=FakePageFetcher(),
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=FakeBookmarkClient(fail=True),
    )

    assert result.status == "bookmark_failed"
    assert "ブックマークを登録できませんでした" in result.comments[-1]
    assert redmine.updated[-1] == (
        123,
        {"notes": result.comments[-1], "assigned_to_id": 7},
    )
    assert "status_id" not in redmine.updated[-1][1]


def test_run_once_dry_run_does_not_update_redmine_or_linkace() -> None:
    redmine = FakeRedmine([{"id": 123}], _issue())
    bookmark_client = FakeBookmarkClient()

    result = run_once(
        config=CONFIG,
        redmine=redmine,
        request_classifier=FakeClassifier(),
        page_fetcher=FakePageFetcher(),
        briefing_summarizer=FakeSummarizer(),
        bookmark_client=bookmark_client,
        dry_run=True,
    )

    assert result.status == "dry_run"
    assert result.dry_run is True
    assert result.briefing == "# エグゼクティブサマリー\n要約本文"
    assert result.bookmark_payload == {
        "url": "https://example.test/article",
        "title": "Article title",
        "description": "# エグゼクティブサマリー\n要約本文",
        "list_id": 10,
    }
    assert redmine.updated == []
    assert bookmark_client.auth_checks == 0
    assert bookmark_client.calls == []


def test_run_once_missing_author_id_fails_without_update() -> None:
    redmine = FakeRedmine(
        [{"id": 123}],
        issue={"id": 123, "author": {"name": "requester"}, "subject": "test"},
    )

    try:
        run_once(
            config=CONFIG,
            redmine=redmine,
            request_classifier=FakeClassifier(),
            page_fetcher=FakePageFetcher(),
            briefing_summarizer=FakeSummarizer(),
            bookmark_client=FakeBookmarkClient(),
        )
    except WorkflowError as exc:
        assert "author did not include an integer id" in str(exc)
    else:
        raise AssertionError("WorkflowError was not raised")

    assert redmine.updated == []
