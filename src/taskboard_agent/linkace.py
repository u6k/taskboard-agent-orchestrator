from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class LinkAceError(RuntimeError):
    """Raised when LinkAce returns an unexpected response."""


class LinkAceAuthenticationError(LinkAceError):
    """Raised when LinkAce rejects the configured API token."""


@dataclass(frozen=True)
class BookmarkResult:
    id: int | None
    url: str | None
    action: str = "created"


@dataclass(frozen=True)
class ExistingBookmark:
    id: int
    url: str
    web_url: str
    list_ids: tuple[int, ...]

    def has_list(self, list_id: int) -> bool:
        return list_id in self.list_ids


class LinkAceClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            transport=transport,
        )

    def add_link(
        self,
        *,
        url: str,
        title: str,
        description: str,
        list_id: int,
    ) -> BookmarkResult:
        existing = self.find_link(url)
        if existing is not None and existing.has_list(1):
            return self._update_link(
                existing.id,
                url=url,
                title=title,
                description=description,
                list_id=list_id,
            )
        if existing is not None:
            return BookmarkResult(
                id=existing.id,
                url=existing.web_url,
                action="already_exists",
            )
        return self._create_link(
            url=url,
            title=title,
            description=description,
            list_id=list_id,
        )

    def check_auth(self) -> None:
        response = self._client.get(
            "/api/v2/links",
            params={"per_page": 1},
        )
        _raise_for_error(response, "failed to authenticate with LinkAce")

    def find_link(self, url: str) -> ExistingBookmark | None:
        response = self._client.get(
            "/api/v2/search/links",
            params={
                "query": url,
                "per_page": -1,
                "search_title": False,
                "search_description": False,
            },
        )
        _raise_for_error(response, "failed to search LinkAce bookmarks")

        for candidate in _response_items(_json_or_empty(response)):
            candidate_id = candidate.get("id")
            if not isinstance(candidate_id, int) or candidate.get("url") != url:
                continue
            detail = self._get_link(candidate_id)
            detail_url = detail.get("url")
            if not isinstance(detail_url, str) or detail_url != url:
                continue
            return ExistingBookmark(
                id=candidate_id,
                url=detail_url,
                web_url=f"{self._base_url}/links/{candidate_id}",
                list_ids=_link_list_ids(detail),
            )
        return None

    def _create_link(
        self,
        *,
        url: str,
        title: str,
        description: str,
        list_id: int,
    ) -> BookmarkResult:
        response = self._client.post(
            "/api/v2/links",
            json=_link_payload(
                url=url,
                title=title,
                description=description,
                list_id=list_id,
            ),
        )
        _raise_for_error(response, "failed to create LinkAce bookmark")

        data = _json_or_empty(response)
        bookmark_id = _find_int(data, "id")
        bookmark_url = _find_str(data, "url")
        if bookmark_id is not None:
            bookmark_url = f"{self._base_url}/links/{bookmark_id}"
        return BookmarkResult(id=bookmark_id, url=bookmark_url, action="created")

    def _update_link(
        self,
        link_id: int,
        *,
        url: str,
        title: str,
        description: str,
        list_id: int,
    ) -> BookmarkResult:
        response = self._client.patch(
            f"/api/v2/links/{link_id}",
            json=_link_payload(
                url=url,
                title=title,
                description=description,
                list_id=list_id,
            ),
        )
        _raise_for_error(response, "failed to update LinkAce bookmark")

        data = _json_or_empty(response)
        bookmark_id = _find_int(data, "id") or link_id
        bookmark_url = _find_str(data, "url")
        if bookmark_id is not None:
            bookmark_url = f"{self._base_url}/links/{bookmark_id}"
        return BookmarkResult(id=bookmark_id, url=bookmark_url, action="updated")

    def _get_link(self, link_id: int) -> dict[str, Any]:
        response = self._client.get(f"/api/v2/links/{link_id}")
        _raise_for_error(response, f"failed to fetch LinkAce bookmark #{link_id}")
        data = _json_or_empty(response)
        link = data.get("data") if isinstance(data.get("data"), dict) else data
        if not isinstance(link, dict):
            raise LinkAceError(
                f"failed to fetch LinkAce bookmark #{link_id}: response JSON was not an object"
            )
        return link


def _raise_for_error(response: httpx.Response, message: str) -> None:
    if response.status_code == 401:
        raise LinkAceAuthenticationError(
            f"{message}: HTTP 401 Unauthenticated. "
            "LINKACE_API_KEY must be a valid LinkAce user or system API token with API access."
        )
    if response.status_code >= 400:
        raise LinkAceError(
            f"{message}: HTTP {response.status_code} {response.text}"
        )


def _json_or_empty(response: httpx.Response) -> dict[str, Any]:
    if response.content == b"":
        return {}
    try:
        data = response.json()
    except ValueError as exc:
        raise LinkAceError("failed to create LinkAce bookmark: response was not JSON") from exc
    if not isinstance(data, dict):
        raise LinkAceError("failed to create LinkAce bookmark: response JSON was not an object")
    return data


def _link_payload(
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
        "lists": [list_id],
        "tags": [],
        "visibility": 1,
        "check_disabled": False,
    }


def _response_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = data.get("data")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _link_list_ids(link: dict[str, Any]) -> tuple[int, ...]:
    lists = link.get("lists")
    if not isinstance(lists, list):
        return ()
    ids: list[int] = []
    for item in lists:
        if isinstance(item, int):
            ids.append(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), int):
            ids.append(item["id"])
    return tuple(ids)


def _find_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if isinstance(value, int):
        return value
    for nested_key in ("data", "link"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            value = nested.get(key)
            if isinstance(value, int):
                return value
    return None


def _find_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if isinstance(value, str):
        return value
    for nested_key in ("data", "link"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            value = nested.get(key)
            if isinstance(value, str):
                return value
    return None
