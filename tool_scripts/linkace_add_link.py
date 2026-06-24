from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from taskboard_agent.tool_loader import ToolRuntimeContext


def create_tool(context: ToolRuntimeContext) -> BaseTool:
    bookmark_client = context.require_service("bookmark_client")
    list_id = context.require_setting("linkace_summarized_list_id")

    @tool(
        parse_docstring=True,
        extras={"risk": "write", "planner_visible": True, "dry_run_safe": True},
    )
    def linkace_add_link(url: str, title: str, description: str) -> dict[str, Any]:
        """指定URL、タイトル、ブリーフィング要約をLinkAceへ登録する。

        Args:
            url: 登録するURL。
            title: 登録するタイトル。
            description: LinkAceに保存する説明または要約。
        """
        payload = {
            "url": url,
            "title": title,
            "description": description,
            "list_id": list_id,
        }
        if context.dry_run:
            return {"ok": True, "dry_run": True, "payload": payload, "bookmark": None}
        try:
            bookmark = bookmark_client.add_link(**payload)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "payload": payload}
        return {"ok": True, "payload": payload, "bookmark": _bookmark_result(bookmark)}

    return linkace_add_link


def _bookmark_result(bookmark: Any) -> dict[str, Any]:
    return {
        "id": bookmark.id,
        "url": bookmark.url,
        "action": bookmark.action,
    }
