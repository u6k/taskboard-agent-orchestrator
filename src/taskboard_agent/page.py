from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

import httpx
import trafilatura


class PageFetchError(RuntimeError):
    """Raised when a web page cannot be fetched or reduced to article text."""


@dataclass(frozen=True)
class PageContent:
    url: str
    title: str
    text: str


class WebPageExtractor:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "taskboard-agent-orchestrator/0.1",
                "Accept": "text/html,application/xhtml+xml",
            },
            transport=transport,
        )

    def fetch(self, url: str) -> PageContent:
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise PageFetchError(f"ページ取得リクエストに失敗しました: {exc}") from exc

        if response.status_code >= 400:
            raise PageFetchError(f"HTTP {response.status_code} が返されました")

        content_type = response.headers.get("content-type", "").lower()
        if content_type and not (
            "text/html" in content_type or "application/xhtml+xml" in content_type
        ):
            raise PageFetchError(f"HTMLではないContent-Typeです: {content_type}")

        html = response.text
        text = trafilatura.extract(
            html,
            url=str(response.url),
            include_comments=False,
            include_tables=True,
        )
        if text is None or text.strip() == "":
            raise PageFetchError("ページ本文を抽出できませんでした")

        title = _extract_title(html) or str(response.url)
        return PageContent(url=str(response.url), title=title.strip(), text=text.strip())


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.parts.append(data)


def _extract_title(html: str) -> str | None:
    parser = _TitleParser()
    parser.feed(html)
    title = " ".join(part.strip() for part in parser.parts if part.strip())
    return title or None
