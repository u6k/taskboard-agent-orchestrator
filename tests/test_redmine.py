from __future__ import annotations

import json

import httpx

from taskboard_agent.redmine import RedmineClient


def test_find_open_issues_uses_ai_assignee_and_open_status() -> None:
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(200, json={"issues": [{"id": 123}]})

    client = RedmineClient(
        "https://redmine.example.test",
        "api-key",
        transport=httpx.MockTransport(handler),
    )

    issues = client.find_open_issues_assigned_to(42)

    assert issues == [{"id": 123}]
    assert seen_request is not None
    params = httpx.QueryParams(seen_request.url.query)
    assert seen_request.method == "GET"
    assert seen_request.url.path == "/issues.json"
    assert params["assigned_to_id"] == "42"
    assert params["status_id"] == "open"
    assert params["limit"] == "1"
    assert params["sort"] == "updated_on:asc"


def test_update_description_note_and_reassign_sends_description_notes_and_assignment() -> None:
    seen_body: dict[str, object] | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body
        seen_body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(204)

    client = RedmineClient(
        "https://redmine.example.test",
        "api-key",
        transport=httpx.MockTransport(handler),
    )

    client.update_description_note_and_reassign(
        123,
        description="# 目的\n整理済み",
        notes="Descriptionを整理しました",
        assigned_to_id=7,
    )

    assert seen_body == {
        "issue": {
            "description": "# 目的\n整理済み",
            "notes": "Descriptionを整理しました",
            "assigned_to_id": 7,
        }
    }


def test_get_issue_requests_journals_and_attachments() -> None:
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(200, json={"issue": {"id": 123, "journals": []}})

    client = RedmineClient(
        "https://redmine.example.test",
        "api-key",
        transport=httpx.MockTransport(handler),
    )

    issue = client.get_issue(123)

    assert issue["id"] == 123
    assert seen_request is not None
    assert httpx.QueryParams(seen_request.url.query)["include"] == "journals,attachments"


def test_download_attachment_uses_api_auth_and_enforces_size() -> None:
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(200, content=b"docx", headers={"Content-Length": "4"})

    client = RedmineClient(
        "https://redmine.example.test/redmine",
        "api-key",
        transport=httpx.MockTransport(handler),
    )

    content = client.download_attachment(
        "https://redmine.example.test/redmine/attachments/download/9/report.docx",
        max_bytes=4,
    )

    assert content == b"docx"
    assert seen_request is not None
    assert seen_request.headers["X-Redmine-API-Key"] == "api-key"


def test_download_attachment_rejects_other_origin_or_base_path() -> None:
    client = RedmineClient("https://redmine.example.test/redmine", "api-key")

    for url in (
        "https://evil.example.test/redmine/attachments/1/a.docx",
        "https://redmine.example.test/other/attachments/1/a.docx",
    ):
        try:
            client.download_attachment(url)
        except Exception as exc:
            assert "attachment URL" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("unsafe attachment URL was accepted")


def test_download_attachment_rejects_declared_oversize() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"large", headers={"Content-Length": "5"})

    client = RedmineClient(
        "https://redmine.example.test",
        "api-key",
        transport=httpx.MockTransport(handler),
    )

    try:
        client.download_attachment(
            "https://redmine.example.test/attachments/download/1/a.docx",
            max_bytes=4,
        )
    except Exception as exc:
        assert "exceeds 4 bytes" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("oversized attachment was accepted")


def test_update_issue_sends_notes_status_and_assignment() -> None:
    seen_body: dict[str, object] | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body
        seen_body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(204)

    client = RedmineClient(
        "https://redmine.example.test",
        "api-key",
        transport=httpx.MockTransport(handler),
    )

    client.update_issue(
        123,
        notes="ブックマークを登録しました。",
        assigned_to_id=7,
        status_id=10,
    )

    assert seen_body == {
        "issue": {
            "notes": "ブックマークを登録しました。",
            "assigned_to_id": 7,
            "status_id": 10,
        }
    }
