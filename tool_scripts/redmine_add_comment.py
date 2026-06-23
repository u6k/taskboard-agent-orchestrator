from __future__ import annotations

from typing import Any

from taskboard_agent.tool_loader import ToolRuntimeContext
from taskboard_agent.tools import ToolSpec


TOOL_SPEC = ToolSpec(
    name="redmine_add_comment",
    description="Redmineチケットへコメントを追加する。",
    parameters={
        "type": "object",
        "properties": {
            "issue_id": {"type": "integer"},
            "notes": {"type": "string"},
        },
        "required": ["issue_id", "notes"],
        "additionalProperties": False,
    },
    risk="write",
    planner_visible=False,
)
DRY_RUN_SAFE = True


def create_handler(context: ToolRuntimeContext) -> Any:
    redmine_client = context.require_service("redmine_client")

    def handle(*, issue_id: int, notes: str) -> dict[str, Any]:
        payload = {"issue_id": issue_id, "notes": notes}
        if context.dry_run:
            return {"ok": True, "dry_run": True, "payload": payload}
        try:
            redmine_client.update_issue(issue_id, notes=notes)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "payload": payload}
        return {"ok": True, "payload": payload}

    return handle
