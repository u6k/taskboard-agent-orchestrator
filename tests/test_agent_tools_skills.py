from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from taskboard_agent.agent import FunctionCallingAgent
from taskboard_agent.llm import LLMResponse, LLMToolCall
from taskboard_agent.skills import SkillRegistry
from taskboard_agent.tool_loader import ToolRuntimeContext, ToolScriptCatalog
from taskboard_agent.tools import (
    ToolRegistry,
    ToolRegistryError,
    ToolSpec,
    parse_tool_arguments,
)


def test_tool_registry_executes_registered_tool_with_schema_validation() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="Echo text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        lambda text: {"text": text},
    )

    result = registry.execute("echo", {"text": "hello"})

    assert result.content == {"text": "hello"}
    assert registry.litellm_tools() == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def test_tool_registry_blocks_write_tools_without_write_permission() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="write_note",
            description="Write a note.",
            parameters={"type": "object", "properties": {}, "required": []},
            risk="write",
        ),
        lambda: {"ok": True},
    )

    with pytest.raises(ToolRegistryError, match="allow_writes"):
        registry.execute("write_note", {})


def test_parse_tool_arguments_rejects_non_object_json() -> None:
    with pytest.raises(ToolRegistryError, match="JSON object"):
        parse_tool_arguments("echo", '["bad"]')


def test_skill_registry_loads_front_matter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "sample"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                'name: "sample"',
                'description: "Sample skill."',
                "required_tools:",
                "  - echo",
                'risk_level: "read"',
                "---",
                "",
                "# 手順",
                "echoを呼ぶ。",
            ]
        ),
        encoding="utf-8",
    )

    skills = SkillRegistry(tmp_path)

    assert skills.summaries() == [
        {
            "name": "sample",
            "description": "Sample skill.",
            "required_tools": ["echo"],
            "risk_level": "read",
        }
    ]
    assert skills.get("sample").body == "# 手順\nechoを呼ぶ。"


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            assert tools is not None
            return LLMResponse(
                content="",
                tool_calls=(
                    LLMToolCall(
                        id="call-1",
                        name="echo",
                        arguments='{"text": "hello"}',
                    ),
                ),
            )
        assert messages[-1]["role"] == "tool"
        return LLMResponse(content="done")


def test_function_calling_agent_executes_tool_calls_until_final_response() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="Echo text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        lambda text: {"text": text},
    )

    result = FunctionCallingAgent(llm=FakeLLM(), tools=registry).run(
        [{"role": "user", "content": "echo hello"}]
    )

    assert result.final_text == "done"
    assert result.stopped_reason == "final"
    assert result.tool_results[0].content == {"text": "hello"}


def test_function_calling_agent_can_return_immediately_after_selected_tool() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="Echo text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        lambda text: {"text": text},
    )
    llm = FakeLLM()

    result = FunctionCallingAgent(llm=llm, tools=registry).run(
        [{"role": "user", "content": "echo hello"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "ignored", "strict": True, "schema": {"type": "object"}},
        },
        return_after_tool_names={"echo"},
    )

    assert llm.calls == 1
    assert result.final_text == ""
    assert result.stopped_reason == "tool_result"
    assert result.tool_results[0].content == {"text": "hello"}


def test_function_calling_agent_emits_intermediate_llm_response() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="Echo text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        lambda text: {"text": text},
    )
    responses: list[LLMResponse] = []

    FunctionCallingAgent(llm=FakeLLM(), tools=registry).run(
        [{"role": "user", "content": "echo hello"}],
        on_llm_response=responses.append,
    )

    assert len(responses) == 1
    assert responses[0].tool_calls[0].name == "echo"


class StructuredToolLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "response_format": response_format,
            }
        )
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                tool_calls=(
                    LLMToolCall(
                        id="call-1",
                        name="echo",
                        arguments='{"text": "hello"}',
                    ),
                ),
            )
        if len(self.calls) == 2:
            return LLMResponse(content="echoを実行しました。")
        return LLMResponse(content='{"status":"processed","notes":"done"}')


def test_function_calling_agent_separates_tools_from_structured_output() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="Echo text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        lambda text: {"text": text},
    )
    llm = StructuredToolLLM()
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "result", "strict": True, "schema": {"type": "object"}},
    }

    result = FunctionCallingAgent(llm=llm, tools=registry).run(
        [{"role": "user", "content": "echo hello"}],
        response_format=response_format,
    )

    assert result.final_text == '{"status":"processed","notes":"done"}'
    assert len(llm.calls) == 3
    assert all(call["response_format"] is None for call in llm.calls[:2])
    assert all(call["tools"] is not None for call in llm.calls[:2])
    assert llm.calls[2]["tools"] is None
    assert llm.calls[2]["response_format"] == response_format


def test_tool_script_catalog_loads_tool_by_name(tmp_path: Path) -> None:
    (tmp_path / "echo.py").write_text(
        "\n".join(
            [
                "from taskboard_agent.tools import ToolSpec",
                "TOOL_SPEC = ToolSpec(",
                '    name="echo",',
                '    description="Echo text.",',
                '    parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},',
                ")",
                "def create_handler(context):",
                "    def handle(*, text):",
                "        return {'text': text, 'prefix': context.require_setting('prefix')}",
                "    return handle",
            ]
        ),
        encoding="utf-8",
    )

    catalog = ToolScriptCatalog(
        tmp_path,
        ToolRuntimeContext(services={}, settings={"prefix": "ok"}),
    )

    registry = catalog.registry_for(("echo",))

    assert catalog.summaries() == [
        {
            "name": "echo",
            "description": "Echo text.",
            "risk": "read",
        }
    ]
    assert registry.execute("echo", {"text": "hello"}).content == {
        "text": "hello",
        "prefix": "ok",
    }


def test_tool_script_catalog_hides_internal_tool_from_planner(tmp_path: Path) -> None:
    (tmp_path / "internal.py").write_text(
        "\n".join(
            [
                "from taskboard_agent.tools import ToolSpec",
                "TOOL_SPEC = ToolSpec(",
                '    name="internal",',
                '    description="Internal tool.",',
                '    parameters={"type": "object", "properties": {}, "required": []},',
                "    planner_visible=False,",
                ")",
                "def create_handler(context):",
                "    return lambda: {'ok': True}",
            ]
        ),
        encoding="utf-8",
    )
    catalog = ToolScriptCatalog(
        tmp_path,
        ToolRuntimeContext(services={}, settings={}),
    )

    assert catalog.summaries() == []
    assert catalog.registry_for(("internal",)).execute("internal", {}).content == {
        "ok": True
    }


def test_tool_script_catalog_allows_dry_run_safe_write_tool(tmp_path: Path) -> None:
    (tmp_path / "write_note.py").write_text(
        "\n".join(
            [
                "from taskboard_agent.tools import ToolSpec",
                "TOOL_SPEC = ToolSpec(",
                '    name="write_note",',
                '    description="Write note.",',
                '    parameters={"type": "object", "properties": {}, "required": []},',
                '    risk="write",',
                ")",
                "DRY_RUN_SAFE = True",
                "def create_handler(context):",
                "    return lambda: {'dry_run': context.dry_run}",
            ]
        ),
        encoding="utf-8",
    )

    catalog = ToolScriptCatalog(
        tmp_path,
        ToolRuntimeContext(services={}, settings={}, dry_run=True),
    )

    registry = catalog.registry_for(("write_note",))

    assert registry.execute("write_note", {}).content == {"dry_run": True}
