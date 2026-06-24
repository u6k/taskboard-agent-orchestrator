from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from taskboard_agent.tool_loader import ToolRuntimeContext


SOURCE_LIST_ID = 1


def create_tool(context: ToolRuntimeContext) -> BaseTool:
    bookmark_client = context.require_service("bookmark_client")

    @tool(
        parse_docstring=True,
        extras={"risk": "read", "planner_visible": True},
    )
    def linkace_find_link(url: str) -> dict[str, Any]:
        """指定URLがLinkAceに登録済みか確認する。

        Args:
            url: LinkAce内で検索するURL。
        """
        if context.dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "skipped": True,
                "found": False,
                "url": url,
            }
        try:
            bookmark = bookmark_client.find_link(url)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": url}
        return {
            "ok": True,
            "found": bookmark is not None,
            "url": url,
            "bookmark": _existing_bookmark(bookmark) if bookmark else None,
        }

    return linkace_find_link


def _existing_bookmark(bookmark: Any) -> dict[str, Any]:
    return {
        "id": bookmark.id,
        "url": bookmark.url,
        "web_url": bookmark.web_url,
        "list_ids": list(bookmark.list_ids),
        "has_source_list": bookmark.has_list(SOURCE_LIST_ID),
    }
