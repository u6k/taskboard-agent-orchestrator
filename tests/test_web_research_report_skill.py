from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool

from taskboard_agent.skill_runtime import ScriptedSkillRunner
from taskboard_agent.skills import SkillRegistry
from taskboard_agent.tool_loader import ToolRuntimeContext, ToolScriptCatalog
from taskboard_agent.tools import is_planner_visible, tool_risk


ROOT = Path(__file__).parents[1]
SKILLS_ROOT = ROOT / "skills"


def _tools(handlers: dict[str, Callable[..., dict[str, Any]]]) -> list[BaseTool]:
    return [
        StructuredTool.from_function(
            handler,
            name=name,
            description=name,
            args_schema=_test_tool_schema(),
            infer_schema=False,
            extras={"risk": "read", "planner_visible": False},
        )
        for name, handler in handlers.items()
    ]


def _test_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "issue_json": {"type": "string"},
            "research_plan_json": {"type": "string"},
            "collected_results_json": {"type": "string"},
            "evaluation_history_json": {"type": "string"},
            "coverage_json": {"type": "string"},
            "search_log_json": {"type": "string"},
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
            "page_text_max_chars": {"type": "integer"},
            "region": {"type": "string"},
            "safesearch": {"type": "string"},
            "timelimit": {"type": ["string", "null"]},
        },
    }


def _report(topic: str = "生成AI") -> str:
    return "\n".join(
        [
            f"# リサーチレポート：{topic}",
            "",
            "## 1. エグゼクティブサマリー",
            "要約です。",
            "",
            "## 2. 導入",
            "導入です。",
            "",
            "## 3. 調査概要",
            "概要です。",
            "",
            "## 4. 本論",
            "本論です。",
            "",
            "## 5. 考察と課題",
            "考察です。",
            "",
            "## 6. 結論と次のステップ",
            "結論です。",
            "",
            "## 7. ソース一覧",
            "| ID | 種別 | サイト名 | 記事・資料タイトル | 公開日 | URL | 主な用途 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "| S1 | 公式 | Example | Example | 不明 | https://example.test/a | 概要 |",
        ]
    )


def _search_result(query: str) -> dict[str, Any]:
    suffix = "a" if "市場" in query else "b"
    return {
        "ok": True,
        "query": query,
        "search_results": [
            {
                "rank": 1,
                "title": f"記事{suffix}",
                "url": f"https://example.test/{suffix}",
                "snippet": "概要",
            }
        ],
        "pages": [
            {
                "rank": 1,
                "url": f"https://example.test/{suffix}",
                "final_url": f"https://example.test/{suffix}",
                "title": f"記事{suffix}",
                "text": "本文",
                "text_truncated": False,
                "fetch_ok": True,
                "error": None,
            }
        ],
        "partial": False,
    }


def _run(
    handlers: dict[str, Callable[..., dict[str, Any]]],
    *,
    subject: str = "調査",
    description: str = "調査テーマ: 生成AI\n調査目的: 事業機会の探索",
):
    skill = SkillRegistry(SKILLS_ROOT).get("web-research-report")
    return ScriptedSkillRunner(skill=skill, tools=_tools(handlers)).run(
        issue={"id": 123, "subject": subject, "description": description},
    )


def test_web_research_report_runs_search_evaluation_loop_and_outputs_markdown() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def plan_web_research(issue_json: str) -> dict[str, Any]:
        calls.append(("plan_web_research", {"issue_json": issue_json}))
        return {
            "ok": True,
            "plan": {
                "topic": "生成AI",
                "purpose": "事業機会の探索",
                "audience": "SNS・ブログ上の不特定多数",
                "target_period": "特に指定なし",
                "target_region": "日本",
                "target_domain": "AI",
                "excluded_scope": "なし",
                "main_question": "生成AIの事業機会は何か",
                "sub_questions": ["市場動向は何か"],
                "initial_queries": [
                    {"query": "生成AI 市場動向", "purpose": "市場動向を確認する"}
                ],
                "notes": [],
            },
        }

    def web_search_pages(**arguments: Any) -> dict[str, Any]:
        calls.append(("web_search_pages", arguments))
        return _search_result(arguments["query"])

    evaluations = [
        {
            "ok": True,
            "evaluation": {
                "sufficient": False,
                "summary": "導入事例の根拠が不足しています。",
                "covered_points": ["市場動向"],
                "missing_points": ["導入事例"],
                "additional_queries": [
                    {"query": "生成AI 導入事例", "purpose": "導入事例を確認する"}
                ],
                "stop_reason": "needs_more_sources",
            },
        },
        {
            "ok": True,
            "evaluation": {
                "sufficient": True,
                "summary": "主要論点を説明できます。",
                "covered_points": ["市場動向", "導入事例"],
                "missing_points": [],
                "additional_queries": [],
                "stop_reason": "sufficient",
            },
        },
    ]

    def evaluate_research_coverage(**arguments: Any) -> dict[str, Any]:
        calls.append(("evaluate_research_coverage", arguments))
        return evaluations.pop(0)

    def compose_research_report(**arguments: Any) -> dict[str, Any]:
        calls.append(("compose_research_report", arguments))
        return {"ok": True, "report": _report()}

    result = _run(
        {
            "plan_web_research": plan_web_research,
            "web_search_pages": web_search_pages,
            "evaluate_research_coverage": evaluate_research_coverage,
            "compose_research_report": compose_research_report,
        }
    )

    assert result.status == "processed"
    assert [name for name, _ in calls] == [
        "plan_web_research",
        "web_search_pages",
        "evaluate_research_coverage",
        "web_search_pages",
        "evaluate_research_coverage",
        "compose_research_report",
    ]
    assert calls[1][1]["query"] == "生成AI 市場動向"
    assert calls[3][1]["query"] == "生成AI 導入事例"
    notes = "\n\n".join(event.notes or "" for event in result.events)
    assert "Web検索を実行します" in notes
    assert "根拠十分性を評価しました" in notes
    assert "導入事例の根拠が不足しています" in notes
    assert result.events[-1].kind == "final_review"
    assert result.events[-1].notes and result.events[-1].notes.startswith("# リサーチレポート：生成AI")


def test_web_research_report_runs_without_purpose_and_defaults_public_audience() -> None:
    observed_issue: dict[str, Any] = {}

    def plan_web_research(issue_json: str) -> dict[str, Any]:
        observed_issue.update(json.loads(issue_json))
        return {
            "ok": True,
            "plan": {
                "topic": "生成AI",
                "purpose": "未指定",
                "audience": "SNS・ブログ上の不特定多数",
                "target_period": "特に指定なし",
                "target_region": "指定なし",
                "target_domain": "指定なし",
                "excluded_scope": "なし",
                "main_question": "生成AIについて何が重要か",
                "sub_questions": ["主要論点は何か"],
                "initial_queries": [{"query": "生成AI", "purpose": "概要を確認する"}],
                "notes": [],
            },
        }

    result = _run(
        {
            "plan_web_research": plan_web_research,
            "web_search_pages": lambda **arguments: _search_result(arguments["query"]),
            "evaluate_research_coverage": lambda **arguments: {
                "ok": True,
                "evaluation": {
                    "sufficient": True,
                    "summary": "十分です。",
                    "covered_points": ["概要"],
                    "missing_points": [],
                    "additional_queries": [],
                    "stop_reason": "sufficient",
                },
            },
            "compose_research_report": lambda **arguments: {
                "ok": True,
                "report": _report(),
            },
        },
        description="調査テーマ: 生成AI",
    )

    assert result.status == "processed"
    assert observed_issue["inferred"]["purpose"] == "未指定"
    assert observed_issue["inferred"]["audience"] == "SNS・ブログ上の不特定多数"


def test_web_research_report_needs_user_when_topic_is_missing() -> None:
    calls: list[str] = []

    def unexpected(**arguments: Any) -> dict[str, Any]:
        calls.append("called")
        return {"ok": True}

    result = _run(
        {
            "plan_web_research": unexpected,
            "web_search_pages": unexpected,
            "evaluate_research_coverage": unexpected,
            "compose_research_report": unexpected,
        },
        subject="調査",
        description="",
    )

    assert result.status == "needs_user"
    assert calls == []
    assert "調査テーマを特定できません" in (result.events[0].notes or "")


def test_web_research_report_skill_and_internal_tools_are_registered() -> None:
    skill = SkillRegistry(SKILLS_ROOT).get("web-research-report")

    assert skill.runner == "run.py"
    assert skill.required_tools == (
        "plan_web_research",
        "web_search_pages",
        "evaluate_research_coverage",
        "compose_research_report",
    )
    assert skill.risk_level == "read"

    class FakeLLM:
        def complete(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("metadata loading should not call LLM")

    catalog = ToolScriptCatalog(
        ROOT / "tool_scripts",
        ToolRuntimeContext(services={"llm": FakeLLM()}, settings={}),
    )
    tools = catalog.tools_for(
        (
            "plan_web_research",
            "evaluate_research_coverage",
            "compose_research_report",
        )
    )

    assert [tool.name for tool in tools] == [
        "plan_web_research",
        "evaluate_research_coverage",
        "compose_research_report",
    ]
    assert all(tool_risk(tool) == "read" for tool in tools)
    assert all(not is_planner_visible(tool) for tool in tools)
