from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from taskboard_agent.tool_loader import ToolRuntimeContext


def create_tool(context: ToolRuntimeContext) -> BaseTool:
    bookmark_client = context.require_service("bookmark_client")

    @tool(
        parse_docstring=True,
        error_on_invalid_docstring=False,
        extras={"risk": "read", "planner_visible": True},
    )
    def linkace_check_auth() -> dict[str, Any]:
        """LinkAce API token authenticationを確認する。"""
        if context.dry_run:
            return {"ok": True, "dry_run": True, "skipped": True}
        try:
            bookmark_client.check_auth()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    return linkace_check_auth
