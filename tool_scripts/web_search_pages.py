from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from taskboard_agent.tool_loader import ToolRuntimeContext


DEFAULT_MAX_RESULTS = 5
DEFAULT_PAGE_TEXT_MAX_CHARS = 6000
logger = logging.getLogger("taskboard_agent.web_search_pages")


def create_tool(context: ToolRuntimeContext) -> BaseTool:
    search_client = context.require_service("search_client")
    page_fetcher = context.require_service("page_fetcher")

    @tool(
        parse_docstring=True,
        extras={"risk": "read", "planner_visible": True},
    )
    def web_search_pages(
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        page_text_max_chars: int = DEFAULT_PAGE_TEXT_MAX_CHARS,
        region: str = "ja-jp",
        safesearch: str = "moderate",
        timelimit: str | None = None,
    ) -> dict[str, Any]:
        """DuckDuckGoでキーワード検索し、検索結果ページ本文も取得する。

        Args:
            query: 検索キーワード。
            max_results: 取得する検索結果件数。
            page_text_max_chars: 各ページ本文の最大文字数。
            region: DuckDuckGo検索の地域指定。
            safesearch: DuckDuckGo検索のセーフサーチ指定。
            timelimit: DuckDuckGo検索の期間指定。指定しない場合はNone。
        """
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

    return web_search_pages


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
