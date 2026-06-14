from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from taskboard_agent.tools import ToolRegistry, ToolRegistryError, ToolSpec


class ToolScriptError(RuntimeError):
    """Raised when a tool script cannot be loaded safely."""


class ToolHandlerFactory(Protocol):
    def __call__(self, context: ToolRuntimeContext) -> Any:
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
                "name": spec.name,
                "description": spec.description,
                "risk": spec.risk,
            }
            for spec in self.specs()
        ]

    def specs(self) -> list[ToolSpec]:
        if not self._root.exists():
            return []
        specs: list[ToolSpec] = []
        for path in sorted(self._root.glob("*.py")):
            module = _load_module(path, path.stem)
            spec = getattr(module, "TOOL_SPEC", None)
            if not isinstance(spec, ToolSpec):
                raise ToolScriptError(f"tool script missing TOOL_SPEC: {path}")
            if spec.name != path.stem:
                raise ToolScriptError(
                    f"tool script name mismatch: expected {path.stem}, got {spec.name}"
                )
            specs.append(spec)
        return specs

    def registry_for(self, tool_names: tuple[str, ...] | list[str]) -> ToolRegistry:
        registry = ToolRegistry()
        for tool_name in tool_names:
            spec, handler = self._load(tool_name)
            registry.register(spec, handler)
        return registry

    def _load(self, tool_name: str) -> tuple[ToolSpec, Any]:
        path = self._root / f"{tool_name}.py"
        if not path.exists():
            raise ToolRegistryError(f"tool script is not registered: {tool_name}")
        module = _load_module(path, tool_name)
        spec = getattr(module, "TOOL_SPEC", None)
        if not isinstance(spec, ToolSpec):
            raise ToolScriptError(f"tool script missing TOOL_SPEC: {path}")
        if spec.name != tool_name:
            raise ToolScriptError(
                f"tool script name mismatch: expected {tool_name}, got {spec.name}"
            )
        if (
            self._context.dry_run
            and spec.risk == "write"
            and getattr(module, "DRY_RUN_SAFE", False) is True
        ):
            spec = replace(spec, risk="read")
        factory = getattr(module, "create_handler", None)
        if not callable(factory):
            raise ToolScriptError(f"tool script missing create_handler: {path}")
        return spec, factory(self._context)


def _load_module(path: Path, tool_name: str) -> ModuleType:
    module_name = f"_taskboard_agent_tool_script_{tool_name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ToolScriptError(f"failed to load tool script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
