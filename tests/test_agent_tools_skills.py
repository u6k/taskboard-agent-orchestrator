from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from taskboard_agent.agent import LangChainAgentRunner
from taskboard_agent.skills import SkillRegistry
from taskboard_agent.tool_loader import ToolRuntimeContext, ToolScriptCatalog
from taskboard_agent.tools import ToolExecutionError, execute_tool


class FakeChatModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def test_langchain_tool_executes_with_inferred_schema_and_policy() -> None:
    @tool(parse_docstring=True, extras={"risk": "read", "planner_visible": True})
    def echo(text: str) -> dict[str, Any]:
        """Echo text.

        Args:
            text: Text to echo.
        """
        return {"text": text}

    result = execute_tool(echo, {"text": "hello"})

    assert result.content == {"text": "hello"}
    assert echo.name == "echo"
    assert echo.description == "Echo text."
    assert echo.args["text"]["description"] == "Text to echo."
    assert echo.extras == {"risk": "read", "planner_visible": True}


def test_write_tool_is_blocked_without_write_permission() -> None:
    @tool(extras={"risk": "write"})
    def write_note() -> dict[str, Any]:
        """Write a note."""
        return {"ok": True}

    with pytest.raises(ToolExecutionError, match="allow_writes"):
        execute_tool(write_note, {})

    assert execute_tool(write_note, {}, allow_writes=True).content == {"ok": True}


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


def test_langchain_agent_runner_executes_tool_calls_until_final_response() -> None:
    @tool
    def echo(text: str) -> dict[str, Any]:
        """Echo text."""
        return {"text": text}

    model = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "echo", "args": {"text": "hello"}}],
            ),
            AIMessage(content="done"),
        ]
    )

    result = LangChainAgentRunner(model=model).run(
        [{"role": "user", "content": "echo hello"}],
        tools=[echo],
    )

    assert result.final_text == "done"
    assert result.stopped_reason == "final"
    assert result.tool_results[0].content == {"text": "hello"}
    assert model.calls == 2


def test_langchain_agent_runner_can_return_after_tool() -> None:
    @tool
    def echo(text: str) -> dict[str, Any]:
        """Echo text."""
        return {"text": text}

    model = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "echo", "args": {"text": "hello"}}],
            ),
            AIMessage(content="should not be used"),
        ]
    )

    result = LangChainAgentRunner(model=model).run(
        [{"role": "user", "content": "echo hello"}],
        tools=[echo],
        return_after_tool_names={"echo"},
    )

    assert result.final_text == ""
    assert result.stopped_reason == "tool_result"
    assert result.tool_results[0].content == {"text": "hello"}
    assert model.calls == 1


def test_langchain_agent_runner_formats_structured_final_response() -> None:
    @tool
    def echo(text: str) -> dict[str, Any]:
        """Echo text."""
        return {"text": text}

    model = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "echo", "args": {"text": "hello"}}],
            ),
            AIMessage(content="natural result"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-2",
                        "name": "result",
                        "args": {"status": "processed", "notes": "done"},
                    }
                ],
            ),
        ]
    )

    result = LangChainAgentRunner(model=model).run(
        [{"role": "user", "content": "echo hello"}],
        tools=[echo],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["status", "notes"],
                },
            },
        },
    )

    assert result.final_text == '{"status": "processed", "notes": "done"}'
    assert model.calls == 3


def test_tool_script_catalog_loads_langchain_tool_by_name(tmp_path: Path) -> None:
    (tmp_path / "echo.py").write_text(
        "\n".join(
            [
                "from typing import Any",
                "from langchain.tools import tool",
                "def create_tool(context):",
                "    @tool(parse_docstring=True, extras={'risk': 'read', 'planner_visible': True})",
                "    def echo(text: str) -> dict[str, Any]:",
                "        '''Echo text.",
                "",
                "        Args:",
                "            text: Text to echo.",
                "        '''",
                "        return {'text': text, 'prefix': context.require_setting('prefix')}",
                "    return echo",
            ]
        ),
        encoding="utf-8",
    )

    catalog = ToolScriptCatalog(
        tmp_path,
        ToolRuntimeContext(services={}, settings={"prefix": "ok"}),
    )
    echo = catalog.tools_for(("echo",))[0]

    assert catalog.summaries() == [
        {
            "name": "echo",
            "description": "Echo text.",
            "risk": "read",
        }
    ]
    assert execute_tool(echo, {"text": "hello"}).content == {
        "text": "hello",
        "prefix": "ok",
    }


def test_tool_script_catalog_hides_internal_tool_from_planner(tmp_path: Path) -> None:
    (tmp_path / "internal.py").write_text(
        "\n".join(
            [
                "from typing import Any",
                "from langchain.tools import tool",
                "def create_tool(context):",
                "    @tool(extras={'risk': 'read', 'planner_visible': False})",
                "    def internal() -> dict[str, Any]:",
                "        '''Internal tool.'''",
                "        return {'ok': True}",
                "    return internal",
            ]
        ),
        encoding="utf-8",
    )
    catalog = ToolScriptCatalog(
        tmp_path,
        ToolRuntimeContext(services={}, settings={}),
    )

    assert catalog.summaries() == []
    assert execute_tool(catalog.tools_for(("internal",))[0], {}).content == {"ok": True}


def test_tool_script_catalog_allows_dry_run_safe_write_tool(tmp_path: Path) -> None:
    (tmp_path / "write_note.py").write_text(
        "\n".join(
            [
                "from typing import Any",
                "from langchain.tools import tool",
                "def create_tool(context):",
                "    @tool(extras={'risk': 'write', 'planner_visible': True, 'dry_run_safe': True})",
                "    def write_note() -> dict[str, Any]:",
                "        '''Write note.'''",
                "        return {'dry_run': context.dry_run}",
                "    return write_note",
            ]
        ),
        encoding="utf-8",
    )

    catalog = ToolScriptCatalog(
        tmp_path,
        ToolRuntimeContext(services={}, settings={}, dry_run=True),
    )
    write_note = catalog.tools_for(("write_note",))[0]

    assert write_note.extras["risk"] == "read"
    assert execute_tool(write_note, {}).content == {"dry_run": True}
