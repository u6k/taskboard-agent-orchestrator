from __future__ import annotations

from typing import Any

from taskboard_agent.tool_loader import ToolRuntimeContext
from taskboard_agent.tools import ToolSpec


TOOL_SPEC = ToolSpec(
    name="linkace_check_auth",
    description="LinkAce API token authenticationを確認する。",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    risk="read",
)


def create_handler(context: ToolRuntimeContext) -> Any:
    bookmark_client = context.require_service("bookmark_client")

    def handle() -> dict[str, Any]:
        if context.dry_run:
            return {"ok": True, "dry_run": True, "skipped": True}
        try:
            bookmark_client.check_auth()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    return handle
