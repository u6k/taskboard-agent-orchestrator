from __future__ import annotations

import logging
from typing import Any

from taskboard_agent.tool_loader import ToolRuntimeContext
from taskboard_agent.tools import ToolSpec


DEFAULT_MAX_RESULTS = 5
DEFAULT_PAGE_TEXT_MAX_CHARS = 6000
logger = logging.getLogger("taskboard_agent.web_search_pages")


TOOL_SPEC = ToolSpec(
    name="web_search_pages",
    description=(
        "DuckDuckGo(ddgs)でキーワード検索し、検索結果ページ本文も取得する。"
        "検索結果と各ページ本文取得の正常/エラー状態を返す。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
            "page_text_max_chars": {"type": "integer"},
            "region": {"type": "string"},
            "safesearch": {"type": "string"},
            "timelimit": {"type": ["string", "null"]},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    risk="read",
)


def create_handler(context: ToolRuntimeContext) -> Any:
    search_client = context.require_service("search_client")
    page_fetcher = context.require_service("page_fetcher")

    def handle(
        *,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        page_text_max_chars: int = DEFAULT_PAGE_TEXT_MAX_CHARS,
        region: str = "ja-jp",
        safesearch: str = "moderate",
        timelimit: str | None = None,
    ) -> dict[str, Any]:
        max_results = _clamp(max_results, minimum=1, maximum=10)
        page_text_max_chars = _clamp(
            page_text_max_chars,
            minimum=1000,
            maximum=20000,
        )
        logger.debug(
            "web_search_pagesを開始します query=%s max_results=%s page_text_max_chars=%s region=%s safesearch=%s timelimit=%s",
            query,
            max_results,
            page_text_max_chars,
            region,
            safesearch,
            timelimit,
        )
        try:
            results = search_client.search(
                query,
                max_results=max_results,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
            )
        except Exception as exc:
            logger.debug("web_search_pagesの検索に失敗しました query=%s error=%s", query, exc)
            return {"ok": False, "error": str(exc), "query": query}
        logger.debug(
            "web_search_pagesの検索結果を取得しました query=%s count=%s results=%s",
            query,
            len(results),
            [
                {
                    "rank": result.rank,
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                }
                for result in results
            ],
        )

        search_results = [
            {
                "rank": result.rank,
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
            }
            for result in results
        ]
        pages = [
            _fetch_page(
                page_fetcher,
                rank=result.rank,
                url=result.url,
                page_text_max_chars=page_text_max_chars,
            )
            for result in results
        ]
        artifact = {
            "type": "web_search_pages",
            "query": query,
            "search_results": search_results,
            "pages": pages,
        }
        return {
            "ok": True,
            "query": query,
            "region": region,
            "safesearch": safesearch,
            "max_results": max_results,
            "search_results": search_results,
            "pages": pages,
            "partial": any(not page["fetch_ok"] for page in pages),
            "context_artifact": artifact,
        }

    return handle


def _fetch_page(
    page_fetcher: Any,
    *,
    rank: int,
    url: str,
    page_text_max_chars: int,
) -> dict[str, Any]:
    logger.debug("web_search_pagesの本文取得を開始します rank=%s url=%s", rank, url)
    try:
        page = page_fetcher.fetch(url)
    except Exception as exc:
        logger.debug(
            "web_search_pagesの本文取得に失敗しました rank=%s url=%s error=%s",
            rank,
            url,
            exc,
        )
        return {
            "rank": rank,
            "url": url,
            "final_url": None,
            "title": None,
            "text": "",
            "text_truncated": False,
            "fetch_ok": False,
            "error": str(exc),
        }

    text = page.text
    truncated = len(text) > page_text_max_chars
    if truncated:
        text = text[:page_text_max_chars]
    logger.debug(
        "web_search_pagesの本文取得に成功しました rank=%s url=%s final_url=%s title=%s text_length=%s truncated=%s",
        rank,
        url,
        page.url,
        page.title,
        len(page.text),
        truncated,
    )
    return {
        "rank": rank,
        "url": url,
        "final_url": page.url,
        "title": page.title,
        "text": text,
        "text_truncated": truncated,
        "fetch_ok": True,
        "error": None,
    }


def _clamp(value: int, *, minimum: int, maximum: int) -> int:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
