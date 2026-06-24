from __future__ import annotations

from pathlib import Path

from taskboard_agent.page import PageContent
from taskboard_agent.tool_loader import ToolRuntimeContext, ToolScriptCatalog
from taskboard_agent.tools import execute_tool
from taskboard_agent.web_search import DuckDuckGoSearchClient, SearchResult


class FakeDDGS:
    calls: list[dict[str, object]] = []

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self) -> "FakeDDGS":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def text(self, query: str, **kwargs: object) -> list[dict[str, str]]:
        self.calls.append({"query": query, **kwargs})
        return [
            {
                "title": "記事A",
                "href": "https://example.test/a",
                "body": "概要A",
            },
            {
                "title": "記事B",
                "href": "https://example.test/b",
                "body": "概要B",
            },
        ]


def test_duckduckgo_search_client_uses_ja_jp_by_default() -> None:
    FakeDDGS.calls.clear()
    client = DuckDuckGoSearchClient(ddgs_factory=FakeDDGS)

    results = client.search("生成AI")

    assert results == (
        SearchResult(
            rank=1,
            title="記事A",
            url="https://example.test/a",
            snippet="概要A",
        ),
        SearchResult(
            rank=2,
            title="記事B",
            url="https://example.test/b",
            snippet="概要B",
        ),
    )
    assert FakeDDGS.calls[0]["region"] == "ja-jp"
    assert FakeDDGS.calls[0]["backend"] == "auto"
    assert FakeDDGS.calls[0]["max_results"] == 5


class FakeSearchClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, **kwargs: object) -> tuple[SearchResult, ...]:
        self.calls.append({"query": query, **kwargs})
        return (
            SearchResult(1, "正常ページ", "https://example.test/ok", "概要1"),
            SearchResult(2, "失敗ページ", "https://example.test/error", "概要2"),
        )


class FakePageFetcher:
    def fetch(self, url: str) -> PageContent:
        if url.endswith("/error"):
            raise RuntimeError("本文を抽出できませんでした")
        return PageContent(
            url="https://example.test/final",
            title="取得タイトル",
            text="本文" * 4000,
        )


def test_web_search_pages_tool_fetches_pages_and_records_partial_errors() -> None:
    search_client = FakeSearchClient()
    catalog = ToolScriptCatalog(
        Path("tool_scripts"),
        ToolRuntimeContext(
            services={
                "search_client": search_client,
                "page_fetcher": FakePageFetcher(),
            },
            settings={},
        ),
    )
    web_search_pages = catalog.tools_for(("web_search_pages",))[0]

    result = execute_tool(
        web_search_pages,
        {"query": "生成AI", "max_results": 99, "page_text_max_chars": 3000},
    ).content

    assert result["ok"] is True
    assert search_client.calls[0]["region"] == "ja-jp"
    assert search_client.calls[0]["max_results"] == 10
    assert result["partial"] is True
    assert result["search_results"][0]["title"] == "正常ページ"
    assert result["pages"][0]["fetch_ok"] is True
    assert result["pages"][0]["text_truncated"] is True
    assert len(result["pages"][0]["text"]) == 3000
    assert result["pages"][1]["fetch_ok"] is False
    assert "本文を抽出できませんでした" in result["pages"][1]["error"]
    assert result["context_artifact"]["type"] == "web_search_pages"
