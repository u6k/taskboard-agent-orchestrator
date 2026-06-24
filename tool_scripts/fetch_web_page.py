from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from taskboard_agent.tool_loader import ToolRuntimeContext


def create_tool(context: ToolRuntimeContext) -> BaseTool:
    page_fetcher = context.require_service("page_fetcher")

    @tool(
        parse_docstring=True,
        extras={"risk": "read", "planner_visible": True},
    )
    def fetch_web_page(url: str) -> dict[str, Any]:
        """指定URLのWebページから最終URL、タイトル、本文を抽出する。

        Args:
            url: 本文を取得するHTTPまたはHTTPSのURL。
        """
        try:
            page = page_fetcher.fetch(url)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": url}
        return {"ok": True, "url": page.url, "title": page.title, "text": page.text}

    return fetch_web_page
