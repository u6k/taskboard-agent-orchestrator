from __future__ import annotations

import json

import httpx
import pytest

from taskboard_agent.linkace import LinkAceAuthenticationError, LinkAceClient, LinkAceError


def test_add_link_posts_linkace_payload_and_returns_web_url() -> None:
    seen_requests: list[httpx.Request] = []
    seen_body: dict[str, object] | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body
        seen_requests.append(request)
        if request.url.path == "/api/v2/search/links":
            return httpx.Response(200, json={"data": []})
        if request.url.path == "/api/v2/links":
            seen_body = json.loads(request.content.decode("utf-8"))
            return httpx.Response(201, json={"data": {"id": 99}})
        return httpx.Response(404)

    client = LinkAceClient(
        "https://linkace.example.test",
        "linkace-token",
        transport=httpx.MockTransport(handler),
    )

    result = client.add_link(
        url="https://example.test/article",
        title="Article title",
        description="要約本文",
        list_id=10,
    )

    assert [request.method for request in seen_requests] == ["GET", "POST"]
    assert seen_requests[0].url.path == "/api/v2/search/links"
    assert httpx.QueryParams(seen_requests[0].url.query)["query"] == "https://example.test/article"
    assert seen_requests[1].url.path == "/api/v2/links"
    assert seen_requests[1].headers["authorization"] == "Bearer linkace-token"
    assert seen_body == {
        "url": "https://example.test/article",
        "title": "Article title",
        "description": "要約本文",
        "lists": [10],
        "tags": [],
        "visibility": 1,
        "check_disabled": False,
    }
    assert result.id == 99
    assert result.url == "https://linkace.example.test/links/99"
    assert result.action == "created"


def test_add_link_updates_existing_same_url_when_it_is_in_source_list() -> None:
    seen_requests: list[httpx.Request] = []
    seen_body: dict[str, object] | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body
        seen_requests.append(request)
        if request.url.path == "/api/v2/search/links":
            return httpx.Response(
                200,
                json={"data": [{"id": 12, "url": "https://example.test/article"}]},
            )
        if request.url.path == "/api/v2/links/12" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "id": 12,
                    "url": "https://example.test/article",
                    "lists": [{"id": 1, "name": "Unsorted"}],
                },
            )
        if request.url.path == "/api/v2/links/12" and request.method == "PATCH":
            seen_body = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"id": 12})
        return httpx.Response(404)

    client = LinkAceClient(
        "https://linkace.example.test",
        "linkace-token",
        transport=httpx.MockTransport(handler),
    )

    result = client.add_link(
        url="https://example.test/article",
        title="Article title",
        description="要約本文",
        list_id=10,
    )

    assert [(request.method, request.url.path) for request in seen_requests] == [
        ("GET", "/api/v2/search/links"),
        ("GET", "/api/v2/links/12"),
        ("PATCH", "/api/v2/links/12"),
    ]
    assert seen_body == {
        "url": "https://example.test/article",
        "title": "Article title",
        "description": "要約本文",
        "lists": [10],
        "tags": [],
        "visibility": 1,
        "check_disabled": False,
    }
    assert result.id == 12
    assert result.url == "https://linkace.example.test/links/12"
    assert result.action == "updated"


def test_add_link_returns_existing_when_same_url_is_not_in_source_list() -> None:
    seen_paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append((request.method, request.url.path))
        if request.url.path == "/api/v2/search/links":
            return httpx.Response(
                200,
                json={"data": [{"id": 12, "url": "https://example.test/article"}]},
            )
        if request.url.path == "/api/v2/links/12" and request.method == "GET":
            return httpx.Response(
                200,
                json={"id": 12, "url": "https://example.test/article", "lists": [{"id": 2}]},
            )
        return httpx.Response(404)

    client = LinkAceClient(
        "https://linkace.example.test",
        "linkace-token",
        transport=httpx.MockTransport(handler),
    )

    result = client.add_link(
        url="https://example.test/article",
        title="Article title",
        description="要約本文",
        list_id=10,
    )

    assert seen_paths == [
        ("GET", "/api/v2/search/links"),
        ("GET", "/api/v2/links/12"),
    ]
    assert result.id == 12
    assert result.url == "https://linkace.example.test/links/12"
    assert result.action == "already_exists"


def test_find_link_returns_existing_bookmark_with_list_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/search/links":
            return httpx.Response(
                200,
                json={"data": [{"id": 12, "url": "https://example.test/article"}]},
            )
        if request.url.path == "/api/v2/links/12":
            return httpx.Response(
                200,
                json={
                    "id": 12,
                    "url": "https://example.test/article",
                    "lists": [{"id": 1}, {"id": 10}],
                },
            )
        return httpx.Response(404)

    client = LinkAceClient(
        "https://linkace.example.test",
        "linkace-token",
        transport=httpx.MockTransport(handler),
    )

    bookmark = client.find_link("https://example.test/article")

    assert bookmark is not None
    assert bookmark.id == 12
    assert bookmark.url == "https://example.test/article"
    assert bookmark.web_url == "https://linkace.example.test/links/12"
    assert bookmark.list_ids == (1, 10)


def test_add_link_raises_on_error_response() -> None:
    client = LinkAceClient(
        "https://linkace.example.test",
        "linkace-token",
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="boom")),
    )

    with pytest.raises(LinkAceError, match="HTTP 500"):
        client.add_link(
            url="https://example.test/article",
            title="Article title",
            description="要約本文",
            list_id=10,
        )


def test_check_auth_raises_clear_error_on_unauthenticated_response() -> None:
    client = LinkAceClient(
        "https://linkace.example.test",
        "linkace-token",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                401,
                json={"message": "Unauthenticated."},
            )
        ),
    )

    with pytest.raises(LinkAceAuthenticationError, match="LINKACE_API_KEY"):
        client.check_auth()
