from __future__ import annotations

from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx


class RedmineError(RuntimeError):
    """Raised when Redmine returns an unexpected response."""


class RedmineClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._base_parts = urlsplit(self._base_url)
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "X-Redmine-API-Key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            transport=transport,
        )

    def find_open_issues_assigned_to(self, assigned_to_id: int) -> list[dict[str, Any]]:
        response = self._client.get(
            "/issues.json",
            params={
                "assigned_to_id": assigned_to_id,
                "status_id": "open",
                "limit": 1,
                "sort": "updated_on:asc",
            },
        )
        data = _json_or_raise(response, "failed to fetch Redmine issues")
        issues = data.get("issues")
        if not isinstance(issues, list):
            raise RedmineError("failed to fetch Redmine issues: missing issues list")
        return issues

    def get_issue(self, issue_id: int) -> dict[str, Any]:
        response = self._client.get(
            f"/issues/{issue_id}.json",
            params={"include": "journals,attachments"},
        )
        data = _json_or_raise(response, f"failed to fetch Redmine issue #{issue_id}")
        issue = data.get("issue")
        if not isinstance(issue, dict):
            raise RedmineError(f"failed to fetch Redmine issue #{issue_id}: missing issue")
        return issue

    def download_attachment(
        self,
        content_url: str,
        *,
        max_bytes: int = 20 * 1024 * 1024,
    ) -> bytes:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        url = urljoin(f"{self._base_url}/", content_url)
        parts = urlsplit(url)
        if (parts.scheme, parts.netloc) != (
            self._base_parts.scheme,
            self._base_parts.netloc,
        ):
            raise RedmineError("attachment URL must use the configured Redmine origin")
        base_path = self._base_parts.path.rstrip("/")
        if base_path and not (
            parts.path == base_path or parts.path.startswith(f"{base_path}/")
        ):
            raise RedmineError("attachment URL must be under the configured Redmine path")

        try:
            with self._client.stream("GET", url) as response:
                if response.status_code >= 400:
                    response.read()
                    raise RedmineError(
                        "failed to download Redmine attachment: "
                        f"HTTP {response.status_code} {response.text}"
                    )
                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError:
                        declared_size = None
                    if declared_size is not None and declared_size > max_bytes:
                        raise RedmineError(
                            f"Redmine attachment exceeds {max_bytes} bytes"
                        )
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise RedmineError(
                            f"Redmine attachment exceeds {max_bytes} bytes"
                        )
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise RedmineError(f"failed to download Redmine attachment: {exc}") from exc
        return b"".join(chunks)

    def update_description_note_and_reassign(
        self,
        issue_id: int,
        *,
        description: str,
        notes: str,
        assigned_to_id: int,
    ) -> None:
        self.update_issue(
            issue_id,
            description=description,
            notes=notes,
            assigned_to_id=assigned_to_id,
        )

    def update_issue(
        self,
        issue_id: int,
        *,
        notes: str | None = None,
        assigned_to_id: int | None = None,
        status_id: int | None = None,
        description: str | None = None,
    ) -> None:
        issue: dict[str, Any] = {}
        if notes is not None:
            issue["notes"] = notes
        if assigned_to_id is not None:
            issue["assigned_to_id"] = assigned_to_id
        if status_id is not None:
            issue["status_id"] = status_id
        if description is not None:
            issue["description"] = description
        if not issue:
            return

        response = self._client.put(
            f"/issues/{issue_id}.json",
            json={"issue": issue},
        )
        if response.status_code >= 400:
            raise RedmineError(
                "failed to update Redmine issue "
                f"#{issue_id}: HTTP {response.status_code} {response.text}"
            )


def _json_or_raise(response: httpx.Response, message: str) -> dict[str, Any]:
    if response.status_code >= 400:
        raise RedmineError(f"{message}: HTTP {response.status_code} {response.text}")
    try:
        data = response.json()
    except ValueError as exc:
        raise RedmineError(f"{message}: response was not JSON") from exc
    if not isinstance(data, dict):
        raise RedmineError(f"{message}: response JSON was not an object")
    return data
