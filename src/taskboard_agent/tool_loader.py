from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from langchain_core.tools import BaseTool

from taskboard_agent.tools import is_dry_run_safe, is_planner_visible, tool_risk


class ToolScriptError(RuntimeError):
    """Raised when a LangChain tool script cannot be loaded safely."""


class ToolFactory(Protocol):
    def __call__(self, context: ToolRuntimeContext) -> BaseTool:
        ...


@dataclass(frozen=True)
class ToolRuntimeContext:
    services: dict[str, Any]
    settings: dict[str, Any]
    dry_run: bool = False

    def require_service(self, name: str) -> Any:
        try:
            return self.services[name]
        except KeyError as exc:
            raise ToolScriptError(f"tool service is not configured: {name}") from exc

    def require_setting(self, name: str) -> Any:
        try:
            return self.settings[name]
        except KeyError as exc:
            raise ToolScriptError(f"tool setting is not configured: {name}") from exc


class ToolScriptCatalog:
    def __init__(self, root: Path, context: ToolRuntimeContext) -> None:
        self._root = root
        self._context = context

    def summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "risk": tool_risk(tool),
            }
            for tool in self.tools()
            if is_planner_visible(tool)
        ]

    def tools(self) -> list[BaseTool]:
        if not self._root.exists():
            return []
        return [self._load(path.stem) for path in sorted(self._root.glob("*.py"))]

    def tools_for(self, tool_names: tuple[str, ...] | list[str]) -> list[BaseTool]:
        return [self._load(tool_name) for tool_name in tool_names]

    def _load(self, tool_name: str) -> BaseTool:
        path = self._root / f"{tool_name}.py"
        if not path.exists():
            raise ToolScriptError(f"tool script is not registered: {tool_name}")
        module = _load_module(path, tool_name)
        factory = getattr(module, "create_tool", None)
        if not callable(factory):
            raise ToolScriptError(f"tool script missing create_tool: {path}")
        tool = factory(self._context)
        if not isinstance(tool, BaseTool):
            raise ToolScriptError(f"create_tool must return BaseTool: {path}")
        if tool.name != tool_name:
            raise ToolScriptError(
                f"tool script name mismatch: expected {tool_name}, got {tool.name}"
            )
        if (
            self._context.dry_run
            and tool_risk(tool) == "write"
            and is_dry_run_safe(tool)
        ):
            tool.extras = {**(tool.extras or {}), "risk": "read"}
        return tool


def _load_module(path: Path, tool_name: str) -> ModuleType:
    module_name = f"_taskboard_agent_tool_script_{tool_name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ToolScriptError(f"failed to load tool script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
