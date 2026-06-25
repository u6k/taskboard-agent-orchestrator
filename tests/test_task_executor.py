from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool, StructuredTool

from taskboard_agent.agent import AgentRunResult
from taskboard_agent.llm import LLMResponse, LLMToolCall
from taskboard_agent.skill_runtime import SkillEvent, SkillExecutionResult
from taskboard_agent.skills import Skill, SkillRegistry
from taskboard_agent.task_executor import (
    GenericTaskRunner,
    LiteLLMTaskPlanner,
    TaskOrchestrator,
    TaskPlan,
    TaskStep,
    TaskPlanningError,
    parse_task_plan,
)
from taskboard_agent.structured_output import task_plan_response_format
from taskboard_agent.tools import ToolExecutionError, ToolExecutionResult


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, Any]]] = []
        self.response_formats: list[dict[str, Any] | None] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.messages.append(messages)
        self.response_formats.append(response_format)
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

    def tools_for(self, tool_names: tuple[str, ...] | list[str]) -> list[BaseTool]:
        self.calls.append({"tool_names": tuple(tool_names)})
        if self.fail:
            raise ToolExecutionError("missing tool script")
        tools: list[BaseTool] = []
        for tool_name in tool_names:
            def run_tool() -> dict[str, Any]:
                """Run test tool."""
                return {"ok": True}

            tools.append(
                StructuredTool.from_function(
                    run_tool,
                    name=tool_name,
                    description="test tool",
                    infer_schema=True,
                    extras={"risk": "read"},
                )
            )
        return tools

    def summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "fetch_web_page",
                "description": "Webページ本文を取得する。",
                "risk": "read",
            }
        ]


class StepToolCatalog:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def tools_for(self, tool_names: tuple[str, ...] | list[str]) -> list[BaseTool]:
        self.calls.append({"tool_names": tuple(tool_names)})
        tools: list[BaseTool] = []
        if "web_search_pages" in tool_names:
            def web_search_pages(query: str) -> dict[str, Any]:
                """Search pages."""
                return {
                    "ok": True,
                    "context_artifact": {
                        "type": "web_search_pages",
                        "query": query,
                        "search_results": [
                            {
                                "rank": 1,
                                "title": "OpenClaw",
                                "url": "https://openclaw.ai/",
                                "snippet": "概要",
                            }
                        ],
                        "pages": [
                            {
                                "rank": 1,
                                "url": "https://openclaw.ai/",
                                "final_url": "https://openclaw.ai/",
                                "title": "OpenClaw",
                                "text": "企業内の自動化に活用できるAIエージェントです。",
                                "text_truncated": False,
                                "fetch_ok": True,
                                "error": None,
                            }
                        ],
                    },
                }

            tools.append(
                StructuredTool.from_function(
                    web_search_pages,
                    name="web_search_pages",
                    description="Search pages.",
                    infer_schema=True,
                    extras={"risk": "read"},
                )
            )
        return tools

    def summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "web_search_pages",
                "description": "Web検索して本文を取得する。",
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
        tools: list[BaseTool] | None = None,
        allow_writes: bool = False,
        approved_tools: set[str] | None = None,
        on_llm_response: Any | None = None,
        response_format: dict[str, Any] | None = None,
        return_after_tool_names: set[str] | None = None,
    ) -> AgentRunResult:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "allow_writes": allow_writes,
                "response_format": response_format,
                "return_after_tool_names": return_after_tool_names,
            }
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


class SearchArtifactSkillAgent:
    def __init__(
        self,
        *,
        final_text: str = '{"status": "processed", "notes": "検索しました。"}',
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.final_text = final_text

    def run(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[BaseTool] | None = None,
        allow_writes: bool = False,
        approved_tools: set[str] | None = None,
        on_llm_response: Any | None = None,
        response_format: dict[str, Any] | None = None,
        return_after_tool_names: set[str] | None = None,
    ) -> AgentRunResult:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "allow_writes": allow_writes,
                "response_format": response_format,
                "return_after_tool_names": return_after_tool_names,
            }
        )
        return AgentRunResult(
            final_text=self.final_text,
            messages=tuple(messages),
            tool_results=(
                ToolExecutionResult(
                    name="web_search_pages",
                    content={
                        "context_artifact": {
                            "type": "web_search_pages",
                            "query": "生成AI",
                            "search_results": [
                                {
                                    "rank": 1,
                                    "title": "正常ページ",
                                    "url": "https://example.test/ok",
                                    "snippet": "概要",
                                },
                                {
                                    "rank": 2,
                                    "title": "失敗ページ",
                                    "url": "https://example.test/error",
                                    "snippet": "",
                                },
                            ],
                            "pages": [
                                {
                                    "rank": 1,
                                    "url": "https://example.test/ok",
                                    "final_url": "https://example.test/final",
                                    "title": "正常ページ",
                                    "text": "本文",
                                    "text_truncated": False,
                                    "fetch_ok": True,
                                    "error": None,
                                },
                                {
                                    "rank": 2,
                                    "url": "https://example.test/error",
                                    "final_url": None,
                                    "title": None,
                                    "text": "",
                                    "text_truncated": False,
                                    "fetch_ok": False,
                                    "error": "HTTP 500",
                                },
                            ],
                        }
                    },
                ),
            ),
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


def _weekly_skill() -> Skill:
    return Skill(
        name="weekly-docx-report-extractor",
        description="添付された週報DOCXを解析する。",
        required_tools=("extract_redmine_docx", "summarize_weekly_docx"),
        risk_level="read",
        path=Path("skills/weekly-docx-report-extractor/SKILL.md"),
        body="手順",
        runner="run.py",
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


def test_parse_task_plan_reads_steps_and_limitations() -> None:
    plan = parse_task_plan(
        '{"decision":"use_tools","reason":"検索後にLLMで提案する",'
        '"skill_name":null,"tool_names":["web_search_pages"],"target_url":null,'
        '"task_input":null,"user_request":null,'
        '"steps":['
        '{"kind":"tool","name":"web_search_pages","purpose":"OpenClawを検索する",'
        '"arguments":{"query":"openclaw"}},'
        '{"kind":"llm","name":null,"purpose":"企業内活用案を提案する","arguments":null}'
        '],'
        '"limitations":["社内規程への適合は未確認"]}'
    )

    assert plan.steps == (
        TaskStep(
            kind="tool",
            name="web_search_pages",
            purpose="OpenClawを検索する",
            arguments={"query": "openclaw"},
        ),
        TaskStep(
            kind="llm",
            name=None,
            purpose="企業内活用案を提案する",
            arguments=None,
        ),
    )
    assert plan.limitations == ("社内規程への適合は未確認",)


def test_parse_task_plan_repairs_empty_step_purpose_from_task_input() -> None:
    plan = parse_task_plan(
        '{"decision":"use_skill","reason":"対象記事を要約して登録する",'
        '"skill_name":"web-briefing-bookmark","tool_names":[],"target_url":null,'
        '"task_input":{"instruction":"指定URLの記事を要約してブックマーク登録する",'
        '"target_url":"https://example.test/article"},'
        '"user_request":null,'
        '"steps":[{"kind":"skill","name":"web-briefing-bookmark","purpose":"",'
        '"arguments":{"target_url":"https://example.test/article"}}],'
        '"limitations":[]}'
    )

    assert plan.steps == (
        TaskStep(
            kind="skill",
            name="web-briefing-bookmark",
            purpose="指定URLの記事を要約してブックマーク登録する",
            arguments={"target_url": "https://example.test/article"},
        ),
    )


def test_parse_task_plan_repairs_empty_step_purpose_from_reason() -> None:
    plan = parse_task_plan(
        '{"decision":"use_tools","reason":"検索結果を取得する",'
        '"skill_name":null,"tool_names":["web_search_pages"],"target_url":null,'
        '"task_input":null,"user_request":null,'
        '"steps":[{"kind":"tool","name":"web_search_pages","purpose":"   ",'
        '"arguments":{"query":"生成AI"}}],'
        '"limitations":[]}'
    )

    assert plan.steps[0].purpose == "検索結果を取得する"


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
    assert "APIで指定された出力構造" in llm.messages[1][-1]["content"]


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
    response_format = llm.response_formats[0]
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert "task_input" in response_format["json_schema"]["schema"]["properties"]
    assert "JSON objectだけ" not in llm.messages[0][1]["content"]
    assert "依頼目的全体と一致するskillがある場合" in llm.messages[0][1]["content"]


def test_task_plan_schema_marks_core_text_fields_non_empty() -> None:
    schema = task_plan_response_format(
        skill_names=["web-briefing-bookmark"],
        tool_names=["web_search_pages"],
    )["json_schema"]["schema"]

    assert schema["properties"]["reason"]["minLength"] == 1
    step_schema = schema["properties"]["steps"]["items"]
    assert step_schema["properties"]["purpose"]["minLength"] == 1


def test_task_planner_retry_prompt_includes_field_specific_repair_hint() -> None:
    invalid = (
        '{"decision":"use_tools","reason":"x",'
        '"skill_name":null,"tool_names":["unknown"],"target_url":null,'
        '"task_input":null,"user_request":null,'
        '"steps":[{"kind":"tool","name":"unknown","purpose":"調べる","arguments":null}],'
        '"limitations":[]}'
    )
    llm = FakeLLM([invalid, invalid, invalid])

    with pytest.raises(TaskPlanningError, match="after 3 attempts"):
        LiteLLMTaskPlanner(llm).plan(
            _issue(),
            [],
            [{"name": "web_search_pages", "description": "検索"}],
        )

    retry_prompt = llm.messages[1][-1]["content"]
    assert "修正指示:" in retry_prompt
    assert "空文字ではなく" in retry_prompt


def test_task_planner_asks_llm_to_validate_explicit_skill_against_request() -> None:
    llm = FakeLLM(
        [
            '{"decision":"use_skill","reason":"明示されたスキルが週報要約依頼と整合する",'
            '"skill_name":"weekly-docx-report-extractor","tool_names":[],'
            '"target_url":null,"task_input":null,"user_request":null}'
        ]
    )

    plan = LiteLLMTaskPlanner(llm).plan(
        {
            "id": 10800,
            "subject": "6/21 週報",
            "description": "weekly-docx-report-extractorスキルを実行してください。",
        },
        [_skill(), _weekly_skill()],
        [],
    )

    assert plan.decision == "use_skill"
    assert plan.skill_name == "weekly-docx-report-extractor"
    assert len(llm.messages) == 1
    planning_prompt = llm.messages[0][1]["content"]
    assert "名前だけで機械的に選ばず" in planning_prompt
    assert "weekly-docx-report-extractorスキルを実行してください" in planning_prompt


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


def test_task_planner_normalizes_display_tool_name_to_registered_name() -> None:
    llm = FakeLLM(
        [
            '{"decision":"use_tools","reason":"明示されたtoolで検索する",'
            '"skill_name":null,"tool_names":["Webページの情報収集 (web_search_pages)"],'
            '"target_url":null,"task_input":null,"user_request":null,'
            '"steps":[{"kind":"tool","name":"Webページの情報収集 (web_search_pages)",'
            '"purpose":"指定キーワードで検索する","arguments":{"query":"パーソナルAIアシスタント"}}],'
            '"limitations":[]}'
        ]
    )

    plan = LiteLLMTaskPlanner(llm).plan(
        _issue(),
        [],
        [{"name": "web_search_pages", "description": "Webページの情報収集"}],
    )

    assert plan.tool_names == ("web_search_pages",)
    assert plan.steps == (
        TaskStep(
            kind="tool",
            name="web_search_pages",
            purpose="指定キーワードで検索する",
            arguments={"query": "パーソナルAIアシスタント"},
        ),
    )


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
    assert skill_agent.calls[0]["response_format"]["type"] == "json_schema"
    assert skill_agent.calls[0]["response_format"]["json_schema"]["strict"] is True
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
    assert skill_agent.calls[0]["response_format"]["type"] == "json_schema"
    assert tool_catalog.calls[0]["tool_names"] == ("fetch_web_page",)
    assert "使用するtool: fetch_web_page" in skill_agent.calls[0]["messages"][1]["content"]
    assert result.events[0].kind == "start"
    assert result.events[0].notes.startswith("ユーザーから依頼された作業は、")
    assert "tool `fetch_web_page`" in result.events[0].notes
    assert "作業を開始します。" in result.events[0].notes
    assert "fetch_web_pageを呼び出します" in result.events[1].notes
    assert "`fetch_web_page` を呼び出します" in result.events[1].notes
    assert result.events[-1].notes == "done"


def test_orchestrator_reports_web_search_page_fetch_status() -> None:
    tool_catalog = FakeToolCatalog()
    skill_agent = SearchArtifactSkillAgent()
    planner = StubPlanner(
        TaskPlan(
            decision="use_tools",
            tool_names=("web_search_pages",),
            reason="キーワード検索依頼",
            task_input={"instruction": "生成AIを検索"},
        )
    )

    result = TaskOrchestrator(
        planner=planner,
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=tool_catalog,
        skill_agent=skill_agent,  # type: ignore[arg-type]
        generic_runner=GenericTaskRunner(FakeLLM([])),
    ).run(issue=_issue())

    assert result.status == "processed"
    assert result.artifacts[0]["type"] == "web_search_pages"
    assert skill_agent.calls[0]["return_after_tool_names"] == {"web_search_pages"}
    assert "`web_search_pages`を使った場合" in skill_agent.calls[0]["messages"][0]["content"]
    assert "検索結果と本文取得結果" in result.events[-1].notes
    assert "本文取得: 正常" in result.events[-1].notes
    assert "本文取得: エラー" in result.events[-1].notes


def test_orchestrator_runs_step_plan_with_tool_then_llm() -> None:
    generic = GenericTaskRunner(FakeLLM(["企業内活用案を提案しました。"]))
    tool_catalog = StepToolCatalog()
    plan = TaskPlan(
        decision="use_tools",
        reason="検索結果を使ってLLMで提案できる",
        tool_names=("Webページの情報収集 (web_search_pages)",),
        steps=(
            TaskStep(
                kind="tool",
                name="Webページの情報収集 (web_search_pages)",
                purpose="OpenClawを検索して本文を取得する",
                arguments={"query": "openclaw"},
            ),
            TaskStep(
                kind="llm",
                purpose="検索結果をもとに企業内活用案を提案する",
            ),
        ),
        limitations=("社内規程への適合は未確認",),
    )

    result = TaskOrchestrator(
        planner=StubPlanner(plan),
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=tool_catalog,  # type: ignore[arg-type]
        skill_agent=FakeSkillAgent(),
        generic_runner=generic,
    ).run(issue=_issue())

    assert result.status == "processed"
    assert tool_catalog.calls[0]["tool_names"] == ("web_search_pages",)
    assert result.artifacts[0]["type"] == "web_search_pages"
    assert "作業ステップ" in result.events[0].notes
    assert "社内規程への適合は未確認" in result.events[0].notes
    assert any("企業内活用案を提案しました" in (event.notes or "") for event in result.events)
    assert "企業内の自動化に活用できるAIエージェント" in str(generic._llm.messages[0])


def test_orchestrator_repairs_missing_web_search_query_from_previous_step() -> None:
    generic = GenericTaskRunner(
        FakeLLM(
            [
                "検索キーワード「パーソナルAIアシスタント ビジネス活用」を用いてWeb検索を実行します。"
            ]
        )
    )
    tool_catalog = StepToolCatalog()
    plan = TaskPlan(
        decision="use_tools",
        reason="検索後に分析する",
        tool_names=("web_search_pages",),
        steps=(
            TaskStep(
                kind="llm",
                purpose="検索キーワードを準備する",
            ),
            TaskStep(
                kind="tool",
                name="web_search_pages",
                purpose="実際にWeb検索を実行し、関連性の高い複数の情報源からテキストデータを取得する。",
                arguments=None,
            ),
        ),
        limitations=("Web検索は取得時点の情報に限定される",),
    )

    result = TaskOrchestrator(
        planner=StubPlanner(plan),
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=tool_catalog,  # type: ignore[arg-type]
        skill_agent=FakeSkillAgent(),
        generic_runner=generic,
    ).run(issue=_issue())

    assert result.status == "processed"
    assert result.artifacts[0]["query"] == "パーソナルAIアシスタント ビジネス活用"
    assert any("不足引数 `query` を文脈から補完" in (event.notes or "") for event in result.events)


def test_orchestrator_recovers_tool_step_after_schema_failure() -> None:
    generic = GenericTaskRunner(FakeLLM([]))
    tool_catalog = StepToolCatalog()
    plan = TaskPlan(
        decision="use_tools",
        reason="検索する",
        tool_names=("web_search_pages",),
        steps=(
            TaskStep(
                kind="tool",
                name="web_search_pages",
                purpose="OpenClawを検索する",
                arguments={"query": "openclaw", "unused": "remove me"},
            ),
        ),
    )

    result = TaskOrchestrator(
        planner=StubPlanner(plan),
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=tool_catalog,  # type: ignore[arg-type]
        skill_agent=FakeSkillAgent(),
        generic_runner=generic,
    ).run(issue=_issue())

    assert result.status == "processed"
    assert result.artifacts[0]["query"] == "openclaw"
    assert any("スキーマにない引数を除去" in (event.notes or "") for event in result.events)
    assert len(tool_catalog.calls) == 1


def test_orchestrator_executes_single_tool_step() -> None:
    tool_catalog = StepToolCatalog()
    plan = TaskPlan(
        decision="use_tools",
        reason="検索する",
        tool_names=("web_search_pages",),
    )
    step = TaskStep(
        kind="tool",
        name="web_search_pages",
        purpose="OpenClawを検索する",
        arguments={"query": "openclaw"},
    )

    execution = TaskOrchestrator(
        planner=StubPlanner(plan),
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=tool_catalog,  # type: ignore[arg-type]
        skill_agent=FakeSkillAgent(),
        generic_runner=GenericTaskRunner(FakeLLM([])),
    ).execute_single_step(
        issue=_issue(),
        plan=plan,
        step=step,
        step_index=1,
    )

    assert execution.index == 1
    assert execution.step == step
    assert execution.result.status == "processed"
    assert execution.should_stop is False
    assert execution.events[0] == SkillEvent("progress", "ステップ 1 を実行しました: OpenClawを検索する")
    assert execution.artifacts[0]["type"] == "web_search_pages"
    assert "ステップ 1 実行結果" in execution.context_messages[0]["content"]
    assert "ステップ 1 成果JSON" in execution.context_messages[1]["content"]


def test_orchestrator_executes_single_skill_step() -> None:
    plan = TaskPlan(decision="use_skill", reason="スキルで処理する")
    step = TaskStep(
        kind="skill",
        name="web-briefing-bookmark",
        purpose="Webページを要約して登録する",
        arguments={"target_url": "https://example.test/article"},
    )

    execution = TaskOrchestrator(
        planner=StubPlanner(plan),
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=FakeToolCatalog(),
        skill_agent=FakeSkillAgent(),
        generic_runner=GenericTaskRunner(FakeLLM([])),
    ).execute_single_step(
        issue=_issue(),
        plan=plan,
        step=step,
        step_index=1,
    )

    assert execution.result.status == "processed"
    assert execution.should_stop is False
    assert execution.events[0].notes == "ステップ 1 を実行しました: Webページを要約して登録する"
    assert any("done" in (event.notes or "") for event in execution.events)


def test_orchestrator_executes_single_llm_step_with_context() -> None:
    generic = GenericTaskRunner(FakeLLM(["単体LLMステップを実行しました。"]))
    plan = TaskPlan(
        decision="no_skill",
        reason="LLMで整理する",
        limitations=("外部確認は未実施",),
    )
    step = TaskStep(kind="llm", purpose="チケット本文を整理する")

    execution = TaskOrchestrator(
        planner=StubPlanner(plan),
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=FakeToolCatalog(),
        skill_agent=FakeSkillAgent(),
        generic_runner=generic,
    ).execute_single_step(
        issue=_issue(),
        plan=plan,
        step=step,
        step_index=2,
        step_context=[{"role": "assistant", "content": "前のステップ結果"}],
    )

    assert execution.result.status == "processed"
    assert execution.events[0].notes == "ステップ 2 を実行しました: チケット本文を整理する"
    assert any("単体LLMステップを実行しました。" in (event.notes or "") for event in execution.events)
    assert "前のステップ結果" in str(generic._llm.messages[0])
    assert "ステップ目的: チケット本文を整理する" in str(generic._llm.messages[0])


def test_orchestrator_executes_single_unavailable_step_as_skipped() -> None:
    plan = TaskPlan(decision="needs_user", reason="実行できない工程がある")
    step = TaskStep(kind="unavailable", purpose="社内システムの権限変更は実行できない")

    execution = TaskOrchestrator(
        planner=StubPlanner(plan),
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=FakeToolCatalog(),
        skill_agent=FakeSkillAgent(),
        generic_runner=GenericTaskRunner(FakeLLM([])),
    ).execute_single_step(
        issue=_issue(),
        plan=plan,
        step=step,
        step_index=3,
    )

    assert execution.result.status == "skipped"
    assert execution.should_stop is False
    assert execution.events == (
        SkillEvent("progress", "未実行の作業 3: 社内システムの権限変更は実行できない"),
    )
    assert execution.artifacts == ()
    assert execution.context_messages == ()


def test_orchestrator_falls_back_when_web_search_final_json_is_empty() -> None:
    tool_catalog = FakeToolCatalog()
    skill_agent = SearchArtifactSkillAgent(final_text="")
    planner = StubPlanner(
        TaskPlan(
            decision="use_tools",
            tool_names=("web_search_pages",),
            reason="キーワード検索依頼",
            task_input={"instruction": "生成AIを検索"},
        )
    )

    result = TaskOrchestrator(
        planner=planner,
        skill_registry=FakeSkillRegistry([_skill()]),  # type: ignore[arg-type]
        tool_catalog=tool_catalog,
        skill_agent=skill_agent,  # type: ignore[arg-type]
        generic_runner=GenericTaskRunner(FakeLLM([])),
    ).run(issue=_issue())

    assert result.status == "processed"
    assert result.artifacts[0]["type"] == "web_search_pages"
    assert "最終JSON生成が空または不正" in result.events[-1].notes
    assert "検索結果と本文取得結果" in result.events[-1].notes
    assert "本文取得: 正常" in result.events[-1].notes
    assert "本文取得: エラー" in result.events[-1].notes


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


def test_generic_runner_executes_revision_with_full_conversation() -> None:
    llm = FakeLLM(
        [
            "## 作業報告\n\n"
            "1. Webページを取得しました。\n"
            "2. ブリーフィング要約を作成しました。\n"
            "3. LinkAceへ登録しました。"
        ]
    )
    runner = GenericTaskRunner(llm)

    result = runner.run(
        issue=_issue(),
        conversation_messages=[
            {"role": "assistant", "content": "ブリーフィング要約を作成しました。"},
            {"role": "assistant", "content": "ブックマークURL: https://bookmark.test/1"},
            {"role": "user", "content": "差し戻しです。作業を説明してください。"},
        ],
        task_plan=TaskPlan(
            decision="no_skill",
            reason="会話履歴だけで説明できる",
        ),
    )

    assert result.status == "processed"
    sent = llm.messages[0]
    assert any("ブリーフィング要約を作成" in message["content"] for message in sent)
    assert any("https://bookmark.test/1" in message["content"] for message in sent)
    assert "直前に作成した再計画" in sent[-1]["content"]
    assert "JSON、コードフェンス" in sent[-1]["content"]
    assert result.events[0].notes is not None
    assert result.events[0].notes.startswith("## 作業報告")


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
