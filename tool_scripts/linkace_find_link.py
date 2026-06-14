from __future__ import annotations

from typing import Any

from taskboard_agent.tool_loader import ToolRuntimeContext
from taskboard_agent.tools import ToolSpec


SOURCE_LIST_ID = 1

TOOL_SPEC = ToolSpec(
    name="linkace_find_link",
    description="指定URLがLinkAceに登録済みか確認する。",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    },
    risk="read",
)


def create_handler(context: ToolRuntimeContext) -> Any:
    bookmark_client = context.require_service("bookmark_client")

    def handle(*, url: str) -> dict[str, Any]:
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

    return handle


def _existing_bookmark(bookmark: Any) -> dict[str, Any]:
    return {
        "id": bookmark.id,
        "url": bookmark.url,
        "web_url": bookmark.web_url,
        "list_ids": list(bookmark.list_ids),
        "has_source_list": bookmark.has_list(SOURCE_LIST_ID),
    }
