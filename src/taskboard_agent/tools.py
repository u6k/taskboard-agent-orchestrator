from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool


ToolRisk = str


class ToolExecutionError(RuntimeError):
    """Raised when a LangChain tool cannot be executed under local policy."""


@dataclass(frozen=True)
class ToolExecutionResult:
    name: str
    content: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.content, ensure_ascii=False)


def tool_risk(tool: BaseTool) -> ToolRisk:
    return str((tool.extras or {}).get("risk") or "read")


def is_planner_visible(tool: BaseTool) -> bool:
    return bool((tool.extras or {}).get("planner_visible", True))


def is_dry_run_safe(tool: BaseTool) -> bool:
    return bool((tool.extras or {}).get("dry_run_safe", False))


def require_tool_policy(
    tool: BaseTool,
    *,
    allow_writes: bool = False,
    approved_tools: set[str] | None = None,
) -> None:
    risk = tool_risk(tool)
    if risk == "write" and not allow_writes:
        raise ToolExecutionError(f"write tool requires allow_writes=True: {tool.name}")
    if risk == "approval_required" and tool.name not in (approved_tools or set()):
        raise ToolExecutionError(f"tool requires human approval: {tool.name}")


def execute_tool(
    tool: BaseTool,
    arguments: dict[str, Any],
    *,
    allow_writes: bool = False,
    approved_tools: set[str] | None = None,
) -> ToolExecutionResult:
    require_tool_policy(
        tool,
        allow_writes=allow_writes,
        approved_tools=approved_tools,
    )
    raw_result = tool.invoke(arguments)
    if isinstance(raw_result, dict):
        content = raw_result
    elif isinstance(raw_result, str):
        content = _loads_result_object(raw_result, tool.name)
    else:
        content = {"result": raw_result}
    return ToolExecutionResult(name=tool.name, content=content)


def require_tools_registered(tools: list[BaseTool], names: list[str] | tuple[str, ...]) -> None:
    registered = {tool.name for tool in tools}
    missing = [name for name in names if name not in registered]
    if missing:
        raise ToolExecutionError(f"skill requires unregistered tools: {', '.join(missing)}")


def tool_by_name(tools: list[BaseTool], name: str) -> BaseTool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise ToolExecutionError(f"unknown tool: {name}")


def _loads_result_object(raw_result: str, tool_name: str) -> dict[str, Any]:
    try:
        data = json.loads(raw_result)
    except json.JSONDecodeError:
        return {"result": raw_result}
    if not isinstance(data, dict):
        return {"result": data}
    return data
