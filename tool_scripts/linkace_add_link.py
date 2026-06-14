from __future__ import annotations

from typing import Any

from taskboard_agent.tool_loader import ToolRuntimeContext
from taskboard_agent.tools import ToolSpec


TOOL_SPEC = ToolSpec(
    name="linkace_add_link",
    description="指定URL、タイトル、ブリーフィング要約をLinkAceへ登録する。",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["url", "title", "description"],
        "additionalProperties": False,
    },
    risk="write",
)
DRY_RUN_SAFE = True


def create_handler(context: ToolRuntimeContext) -> Any:
    bookmark_client = context.require_service("bookmark_client")
    list_id = context.require_setting("linkace_summarized_list_id")

    def handle(*, url: str, title: str, description: str) -> dict[str, Any]:
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

    return handle


def _bookmark_result(bookmark: Any) -> dict[str, Any]:
    return {
        "id": bookmark.id,
        "url": bookmark.url,
        "action": bookmark.action,
    }
