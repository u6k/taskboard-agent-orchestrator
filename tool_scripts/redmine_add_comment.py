from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from taskboard_agent.tool_loader import ToolRuntimeContext


def create_tool(context: ToolRuntimeContext) -> BaseTool:
    redmine_client = context.require_service("redmine_client")

    @tool(
        parse_docstring=True,
        extras={"risk": "write", "planner_visible": False, "dry_run_safe": True},
    )
    def redmine_add_comment(issue_id: int, notes: str) -> dict[str, Any]:
        """Redmineチケットへコメントを追加する。

        Args:
            issue_id: コメントを追加するRedmineチケットID。
            notes: Redmineへ投稿するコメント本文。
        """
        payload = {"issue_id": issue_id, "notes": notes}
        if context.dry_run:
            return {"ok": True, "dry_run": True, "payload": payload}
        try:
            redmine_client.update_issue(issue_id, notes=notes)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "payload": payload}
        return {"ok": True, "payload": payload}

    return redmine_add_comment
