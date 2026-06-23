from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from ddgs import DDGS


logger = logging.getLogger(__name__)


class WebSearchError(RuntimeError):
    """Raised when a keyword search cannot be completed."""


@dataclass(frozen=True)
class SearchResult:
    rank: int
    title: str
    url: str
    snippet: str


class DuckDuckGoSearchClient:
    def __init__(
        self,
        *,
        timeout: float = 10.0,
        ddgs_factory: Any = DDGS,
    ) -> None:
        self._timeout = timeout
        self._ddgs_factory = ddgs_factory

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        region: str = "ja-jp",
        safesearch: str = "moderate",
        timelimit: str | None = None,
    ) -> tuple[SearchResult, ...]:
        query = query.strip()
        if not query:
            raise WebSearchError("検索キーワードが空です")

        logger.debug(
            "ddgs検索を開始します query=%s region=%s safesearch=%s timelimit=%s max_results=%s backend=auto",
            query,
            region,
            safesearch,
            timelimit,
            max_results,
        )
        try:
            with self._ddgs_factory(timeout=self._timeout) as ddgs:
                raw_results = ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch,
                    timelimit=timelimit,
                    max_results=max_results,
                    page=1,
                    backend="auto",
                )
        except Exception as exc:
            logger.debug("ddgs検索に失敗しました query=%s error=%s", query, exc)
            raise WebSearchError(f"DuckDuckGo検索に失敗しました: {exc}") from exc
        logger.debug(
            "ddgs検索結果を取得しました query=%s raw_count=%s raw_results=%s",
            query,
            len(raw_results or []),
            raw_results,
        )

        normalized: list[SearchResult] = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            url = _string(item.get("href") or item.get("url"))
            if not url:
                continue
            normalized.append(
                SearchResult(
                    rank=len(normalized) + 1,
                    title=_string(item.get("title")) or url,
                    url=url,
                    snippet=_string(item.get("body") or item.get("snippet")),
                )
            )
        logger.debug(
            "ddgs検索結果を正規化しました query=%s count=%s results=%s",
            query,
            len(normalized[:max_results]),
            [
                {
                    "rank": result.rank,
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                }
                for result in normalized[:max_results]
            ],
        )
        return tuple(normalized[:max_results])


def _string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()
