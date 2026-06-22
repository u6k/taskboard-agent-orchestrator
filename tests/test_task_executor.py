from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from taskboard_agent.agent import AgentRunResult
from taskboard_agent.llm import LLMResponse, LLMToolCall
from taskboard_agent.skill_runtime import SkillEvent, SkillExecutionResult
from taskboard_agent.skills import Skill, SkillRegistry
from taskboard_agent.task_executor import (
    GenericTaskRunner,
    LiteLLMTaskPlanner,
    TaskOrchestrator,
    TaskPlan,
    TaskPlanningError,
    parse_task_plan,
)
from taskboard_agent.tools import ToolRegistry, ToolRegistryError, ToolSpec


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, Any]]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> LLMResponse:
        self.messages.append(messages)
        return LLMResponse(content=self.responses.pop(0))


class FakeSkillRegistry:
    def __init__(self, skills: list[Skill]) -> None:
        self._skills = skills

    def list(self) -> list[Skill]:
        return self._skills


class FakeToolCatalog:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def registry_for(self, tool_names: tuple[str, ...] | list[str]) -> ToolRegistry:
        self.calls.append({"tool_names": tuple(tool_names)})
        if self.fail:
            raise ToolRegistryError("missing tool script")
        registry = ToolRegistry()
        for tool_name in tool_names:
            registry.register(
                ToolSpec(
                    name=tool_name,
                    description="test tool",
                    parameters={"type": "object", "properties": {}, "required": []},
                ),
                lambda **_kwargs: {"ok": True},
            )
        return registry

    def summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "fetch_web_page",
                "description": "Webページ本文を取得する。",
                "risk": "read",
            }
        ]


class FakeSkillAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: ToolRegistry | None = None,
        allow_writes: bool = False,
        approved_tools: set[str] | None = None,
        on_llm_response: Any | None = None,
    ) -> AgentRunResult:
        self.calls.append(
            {"messages": messages, "tools": tools, "allow_writes": allow_writes}
        )
        if on_llm_response is not None:
            on_llm_response(
                LLMResponse(
                    content="本文を取得するためfetch_web_pageを呼び出します。",
                    tool_calls=(
                        LLMToolCall(
                            id="call-1",
                            name="fetch_web_page",
                            arguments='{"url": "https://example.test/article"}',
                        ),
                    ),
                )
            )
        return AgentRunResult(
            final_text=(
                '{"status": "processed", "notes": "done", '
                '"target_url": "https://example.test/article"}'
            ),
            messages=tuple(messages),
            tool_results=(),
            stopped_reason="final",
        )


def _issue() -> dict[str, Any]:
    return {
        "id": 123,
        "subject": "要約して",
        "description": "https://example.test/article を要約して登録して",
    }


def _skill() -> Skill:
    return Skill(
        name="web-briefing-bookmark",
        description="Webページを要約してブックマークする。",
        required_tools=("fetch_web_page",),
        risk_level="write",
        path=Path("skills/web-briefing-bookmark/SKILL.md"),
        body="手順",
    )


def test_parse_task_plan_reads_use_skill() -> None:
    plan = parse_task_plan(
        '{"decision": "use_skill", "skill_name": "web-briefing-bookmark", '
        '"target_url": "https://example.test/article", "reason": "URL要約依頼", '
        '"task_input": {"target_url": "https://example.test/article"}, '
        '"user_request": null}'
    )

    assert plan == TaskPlan(
        decision="use_skill",
        skill_name="web-briefing-bookmark",
        target_url="https://example.test/article",
        task_input={"target_url": "https://example.test/article"},
        reason="URL要約依頼",
        user_request=None,
    )


def test_parse_task_plan_reads_use_tools() -> None:
    plan = parse_task_plan(
        '{"decision": "use_tools", "skill_name": null, '
        '"tool_names": ["fetch_web_page"], '
        '"target_url": "https://example.test/article", "reason": "本文取得のみ", '
        '"task_input": {"target_url": "https://example.test/article"}, '
        '"user_request": null}'
    )

    assert plan == TaskPlan(
        decision="use_tools",
        skill_name=None,
        tool_names=("fetch_web_page",),
        target_url="https://example.test/article",
        task_input={"target_url": "https://example.test/article"},
        reason="本文取得のみ",
        user_request=None,
    )


def test_parse_task_plan_rejects_invalid_decision() -> None:
    with pytest.raises(TaskPlanningError, match="decision"):
        parse_task_plan('{"decision": "bad", "reason": "x"}')


def test_parse_task_plan_normalizes_quoted_null_for_nullable_fields() -> None:
    plan = parse_task_plan(
        '{"decision": "use_skill", "skill_name": "web-briefing-bookmark", '
        '"tool_names": [], "target_url": "https://example.test/article", '
        '"task_input": "null", "reason": "対象", "user_request": "null"}'
    )

    assert plan.task_input is None
    assert plan.user_request is None


def test_parse_task_plan_rejects_non_null_string_task_input() -> None:
    with pytest.raises(TaskPlanningError, match="object or null"):
        parse_task_plan(
            '{"decision": "use_skill", "skill_name": "web-briefing-bookmark", '
            '"tool_names": [], "target_url": null, "task_input": "bad", '
            '"reason": "対象", "user_request": null}'
        )


def test_task_planner_retries_with_validation_error_and_accepts_correction() -> None:
    llm = FakeLLM(
        [
            '{"decision": "use_skill", "skill_name": "web-briefing-bookmark", '
            '"tool_names": ["fetch_web_page"], '
            '"target_url": "https://example.test/article", '
            '"task_input": "null", "reason": "対象", "user_request": "null"}',
            '{"decision": "use_skill", "skill_name": "web-briefing-bookmark", '
            '"tool_names": [], "target_url": "https://example.test/article", '
            '"task_input": null, "reason": "対象", "user_request": null}',
        ]
    )

    plan = LiteLLMTaskPlanner(llm).plan(
        _issue(),
        [_skill()],
        [{"name": "fetch_web_page", "description": "本文取得"}],
    )

    assert plan.decision == "use_skill"
    assert len(llm.messages) == 2
    assert "tool_names to be an empty array" in llm.messages[1][-1]["content"]
    assert '文字列の "null"' in llm.messages[1][-1]["content"]


def test_task_planner_fails_after_two_correction_retries() -> None:
    invalid = (
        '{"decision": "use_skill", "skill_name": "web-briefing-bookmark", '
        '"tool_names": ["fetch_web_page"], "target_url": null, '
        '"task_input": null, "reason": "対象", "user_request": null}'
    )
    llm = FakeLLM([invalid, invalid, invalid])

    with pytest.raises(TaskPlanningError, match="after 3 attempts"):
        LiteLLMTaskPlanner(llm).plan(
            _issue(),
            [_skill()],
            [{"name": "fetch_web_page", "description": "本文取得"}],
        )

    assert len(llm.messages) == 3


def test_task_planner_includes_available_skills_in_prompt() -> None:
    llm = FakeLLM(
        [
            '{"decision": "use_skill", "skill_name": "web-briefing-bookmark", '
            '"target_url": "https://example.test/article", "reason": "対象", '
            '"task_input": null, "user_request": null}'
        ]
    )

    plan = LiteLLMTaskPlanner(llm).plan(
        _issue(),
        [_skill()],
        [{"name": "fetch_web_page", "description": "Webページ本文を取得する。"}],
    )

    assert plan.skill_name == "web-briefing-bookmark"
    assert "web-briefing-bookmark" in llm.messages[0][1]["content"]
    assert "fetch_web_page" in llm.messages[0][1]["content"]
    assert '"task_input": null' in llm.messages[0][1]["content"]
    assert "文字列は禁止" in llm.messages[0][1]["content"]


def test_task_planner_can_choose_direct_tools_for_narrow_request() -> None:
    issue = {
        "id": 123,
        "subject": "本文取得",
        "description": "https://example.test/article の本文を取得して",
    }
    llm = FakeLLM(
        [
            '{"decision": "use_tools", "skill_name": null, '
            '"tool_names": ["fetch_web_page"], '
            '"target_url": "https://example.test/article", '
            '"reason": "本文取得だけでブックマーク登録は依頼されていない", '
            '"task_input": null, "user_request": null}'
        ]
    )

    plan = LiteLLMTaskPlanner(llm).plan(
        issue,
        [_skill()],
        [{"name": "fetch_web_page", "description": "Webページ本文を取得する。"}],
    )

    assert plan.decision == "use_tools"
    assert plan.tool_names == ("fetch_web_page",)
    assert plan.skill_name is None


def test_orchestrator_runs_selected_skill() -> None:
    tool_catalog = FakeToolCatalog()
    skill_agent = FakeSkillAgent()
    planner = StubPlanner(
        TaskPlan(
            decision="use_skill",
            skill_name="web-briefing-bookmark",
            target_url="https://example.test/article",
            reason="対象",
        )
    )

    result = TaskOrchestrator(
        planner=planner,
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=tool_catalog,
        skill_agent=skill_agent,
        generic_runner=GenericTaskRunner(FakeLLM([])),
    ).run(issue=_issue())

    assert result.status == "processed"
    assert tool_catalog.calls[0]["tool_names"] == ("fetch_web_page",)
    assert '"target_url": "https://example.test/article"' in skill_agent.calls[0]["messages"][1]["content"]
    assert result.events[0].kind == "start"
    assert result.events[0].notes.startswith("ユーザーから依頼された作業は、")
    assert "作業を開始します。" in result.events[0].notes
    assert "fetch_web_pageを呼び出します" in result.events[1].notes
    assert "`fetch_web_page` を呼び出します" in result.events[1].notes
    assert result.events[-1].notes == "done"


def test_orchestrator_runs_selected_tools_without_skill() -> None:
    tool_catalog = FakeToolCatalog()
    skill_agent = FakeSkillAgent()
    planner = StubPlanner(
        TaskPlan(
            decision="use_tools",
            tool_names=("fetch_web_page",),
            target_url="https://example.test/article",
            reason="本文取得のみ",
        )
    )

    result = TaskOrchestrator(
        planner=planner,
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=tool_catalog,
        skill_agent=skill_agent,
        generic_runner=GenericTaskRunner(FakeLLM([])),
    ).run(issue=_issue())

    assert result.status == "processed"
    assert tool_catalog.calls[0]["tool_names"] == ("fetch_web_page",)
    assert "使用するtool: fetch_web_page" in skill_agent.calls[0]["messages"][1]["content"]
    assert result.events[0].kind == "start"
    assert result.events[0].notes.startswith("ユーザーから依頼された作業は、")
    assert "tool `fetch_web_page`" in result.events[0].notes
    assert "作業を開始します。" in result.events[0].notes
    assert "fetch_web_pageを呼び出します" in result.events[1].notes
    assert "`fetch_web_page` を呼び出します" in result.events[1].notes
    assert result.events[-1].notes == "done"


def test_orchestrator_returns_needs_user_when_skill_is_missing() -> None:
    planner = StubPlanner(
        TaskPlan(
            decision="use_skill",
            skill_name="unknown-skill",
            target_url="https://example.test/article",
            reason="対象",
        )
    )

    result = TaskOrchestrator(
        planner=planner,
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=FakeToolCatalog(),
        skill_agent=FakeSkillAgent(),
        generic_runner=GenericTaskRunner(FakeLLM([])),
    ).run(issue=_issue())

    assert result.status == "needs_user"
    assert result.events[0].kind == "final_review"
    assert "登録されていません" in result.events[0].notes


def test_orchestrator_runs_no_skill_fallback() -> None:
    planner = StubPlanner(TaskPlan(decision="no_skill", reason="文面だけで可能"))
    generic = GenericTaskRunner(
        FakeLLM(['{"status": "completed", "notes": "文面を整理しました。"}'])
    )

    result = TaskOrchestrator(
        planner=planner,
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=FakeToolCatalog(),
        skill_agent=FakeSkillAgent(),
        generic_runner=generic,
    ).run(issue=_issue())

    assert result.status == "processed"
    assert result.events[0].kind == "start"
    assert result.events[0].notes.startswith("ユーザーから依頼された作業は、")
    assert "作業を開始します。" in result.events[0].notes
    assert result.events[1].kind == "final_review"
    assert result.events[1].notes == "文面を整理しました。"


def test_orchestrator_returns_needs_user_for_unclear_request() -> None:
    planner = StubPlanner(
        TaskPlan(
            decision="needs_user",
            reason="依頼が曖昧",
            user_request="何を作業すべきか追記してください。",
        )
    )

    result = TaskOrchestrator(
        planner=planner,
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=FakeToolCatalog(),
        skill_agent=FakeSkillAgent(),
        generic_runner=GenericTaskRunner(FakeLLM([])),
    ).run(issue=_issue())

    assert result.status == "needs_user"
    assert result.events[0] == SkillEvent(
        "final_review",
        "何を作業すべきか追記してください。",
    )


def test_orchestrator_returns_needs_user_when_required_tool_is_missing() -> None:
    planner = StubPlanner(
        TaskPlan(
            decision="use_skill",
            skill_name="web-briefing-bookmark",
            target_url="https://example.test/article",
            reason="対象",
        )
    )

    result = TaskOrchestrator(
        planner=planner,
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=FakeToolCatalog(fail=True),
        skill_agent=FakeSkillAgent(),
        generic_runner=GenericTaskRunner(FakeLLM([])),
    ).run(issue=_issue())

    assert result.status == "needs_user"
    assert "必要なtoolが不足" in result.events[0].notes


class StubPlanner:
    def __init__(self, plan: TaskPlan) -> None:
        self.plan_value = plan

    def plan(
        self,
        issue: dict[str, Any],
        skills: list[Skill],
        tools: list[dict[str, Any]],
    ) -> TaskPlan:
        return self.plan_value
