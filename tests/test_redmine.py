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
