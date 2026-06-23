from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


ToolRisk = Literal["read", "write", "approval_required"]


class ToolRegistryError(RuntimeError):
    """Raised when a tool cannot be registered or executed safely."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    risk: ToolRisk = "read"
    planner_visible: bool = True

    def as_litellm_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolExecutionResult:
    name: str
    content: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.content, ensure_ascii=False)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register(self, spec: ToolSpec, handler: Callable[..., Any]) -> None:
        if spec.name in self._tools:
            raise ToolRegistryError(f"tool is already registered: {spec.name}")
        if spec.parameters.get("type") != "object":
            raise ToolRegistryError(f"tool parameters must be an object schema: {spec.name}")
        self._tools[spec.name] = (spec, handler)

    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def litellm_tools(self) -> list[dict[str, Any]]:
        return [spec.as_litellm_tool() for spec in self.specs()]

    def require_registered(self, names: list[str] | tuple[str, ...]) -> None:
        missing = [name for name in names if name not in self._tools]
        if missing:
            raise ToolRegistryError(f"skill requires unregistered tools: {', '.join(missing)}")

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        allow_writes: bool = False,
        approved_tools: set[str] | None = None,
    ) -> ToolExecutionResult:
        if name not in self._tools:
            raise ToolRegistryError(f"unknown tool: {name}")

        spec, handler = self._tools[name]
        if spec.risk == "write" and not allow_writes:
            raise ToolRegistryError(f"write tool requires allow_writes=True: {name}")
        if spec.risk == "approval_required" and name not in (approved_tools or set()):
            raise ToolRegistryError(f"tool requires human approval: {name}")

        validated = _validate_arguments(name, spec.parameters, arguments)
        result = handler(**validated)
        if isinstance(result, dict):
            content = result
        else:
            content = {"result": result}
        return ToolExecutionResult(name=name, content=content)


def parse_tool_arguments(name: str, raw_arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    try:
        data = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ToolRegistryError(f"tool arguments were not valid JSON for {name}") from exc
    if not isinstance(data, dict):
        raise ToolRegistryError(f"tool arguments must be a JSON object for {name}")
    return data


def _validate_arguments(
    tool_name: str,
    schema: dict[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    if not isinstance(required, list):
        required = []

    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolRegistryError(
            f"tool arguments missing required fields for {tool_name}: {', '.join(missing)}"
        )

    additional_allowed = schema.get("additionalProperties", True) is not False
    if not additional_allowed:
        unknown = [key for key in arguments if key not in properties]
        if unknown:
            raise ToolRegistryError(
                f"tool arguments included unknown fields for {tool_name}: {', '.join(unknown)}"
            )

    for key, value in arguments.items():
        property_schema = properties.get(key)
        if isinstance(property_schema, dict):
            _validate_type(tool_name, key, value, property_schema.get("type"))
    return arguments


def _validate_type(tool_name: str, key: str, value: Any, expected: Any) -> None:
    if expected is None:
        return
    expected_types = expected if isinstance(expected, list) else [expected]
    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, int | float) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "null": lambda item: item is None,
    }
    if any(checks.get(item, lambda _value: True)(value) for item in expected_types):
        return
    expected_text = "|".join(str(item) for item in expected_types)
    raise ToolRegistryError(
        f"tool argument {key} for {tool_name} must be {expected_text}"
    )
