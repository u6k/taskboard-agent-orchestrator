from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from taskboard_agent.skill_runtime import ScriptedSkillRunner
from taskboard_agent.skills import SkillRegistry
from taskboard_agent.tool_loader import ToolRuntimeContext, ToolScriptCatalog
from taskboard_agent.tools import ToolRegistry, ToolSpec


SKILLS_ROOT = Path(__file__).parents[1] / "skills"


def _registry(handlers: dict[str, Callable[..., dict[str, Any]]]) -> ToolRegistry:
    registry = ToolRegistry()
    for name, handler in handlers.items():
        registry.register(
            ToolSpec(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
            ),
            handler,
        )
    return registry


def _run(
    handlers: dict[str, Callable[..., dict[str, Any]]],
    *,
    description: str = "https://example.test/article を要約して登録して",
    task_input: dict[str, Any] | None = None,
    dry_run: bool = False,
):
    skill = SkillRegistry(SKILLS_ROOT).get("web-briefing-bookmark")
    return ScriptedSkillRunner(skill=skill, tools=_registry(handlers)).run(
        issue={"id": 123, "subject": "要約", "description": description},
        task_input=task_input,
        dry_run=dry_run,
    )


def test_scripted_skill_runs_steps_in_order_and_posts_full_summary() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def record(name: str, result: dict[str, Any]):
        def handle(**arguments: Any) -> dict[str, Any]:
            calls.append((name, arguments))
            return result

        return handle

    def comment(**arguments: Any) -> dict[str, Any]:
        raise AssertionError("スキルがRedmineへ直接コメントしました")

    result = _run(
        {
            "linkace_find_link": record(
                "linkace_find_link", {"ok": True, "found": False, "bookmark": None}
            ),
            "fetch_web_page": record(
                "fetch_web_page",
                {
                    "ok": True,
                    "url": "https://example.test/final",
                    "title": "Article",
                    "text": "本文",
                },
            ),
            "summarize_briefing": record(
                "summarize_briefing", {"ok": True, "briefing": "要約全文"}
            ),
            "linkace_add_link": record(
                "linkace_add_link",
                {
                    "ok": True,
                    "payload": {"url": "https://example.test/final"},
                    "bookmark": {
                        "url": "https://linkace.example.test/links/99",
                        "action": "created",
                    },
                },
            ),
            "redmine_add_comment": comment,
        }
    )

    assert result.status == "processed"
    assert result.target_url == "https://example.test/final"
    assert result.briefing == "要約全文"
    assert result.events[-1].kind == "final_review"
    assert result.events[-1].notes == "作業が終了しました。"
    assert [name for name, _ in calls] == [
        "linkace_find_link",
        "fetch_web_page",
        "summarize_briefing",
        "linkace_add_link",
    ]
    comments = [event.notes or "" for event in result.events]
    assert "https://example.test/final" in comments[0]
    assert "要約全文" in comments[1]
    assert "登録しました" in comments[2]


def test_scripted_skill_stops_for_existing_bookmark_outside_source_list() -> None:
    calls: list[str] = []
    comments: list[str] = []

    def find(**arguments: Any) -> dict[str, Any]:
        calls.append("find")
        return {
            "ok": True,
            "found": True,
            "bookmark": {
                "has_source_list": False,
                "web_url": "https://linkace.example.test/links/12",
            },
        }

    def comment(**arguments: Any) -> dict[str, Any]:
        calls.append("comment")
        comments.append(arguments["notes"])
        return {"ok": True}

    def unexpected(**arguments: Any) -> dict[str, Any]:
        raise AssertionError("後続toolが呼ばれました")

    result = _run(
        {
            "linkace_find_link": find,
            "fetch_web_page": unexpected,
            "summarize_briefing": unexpected,
            "linkace_add_link": unexpected,
            "redmine_add_comment": comment,
        }
    )

    assert result.status == "already_done"
    assert result.bookmark_url == "https://linkace.example.test/links/12"
    assert calls == ["find"]
    assert "要約、登録、更新は行いません" in (result.events[0].notes or "")
    assert result.events[-1].notes == "作業が終了しました。"


def test_scripted_skill_stops_after_step_failure_and_comments_reason() -> None:
    calls: list[str] = []
    comments: list[str] = []

    def find(**arguments: Any) -> dict[str, Any]:
        calls.append("find")
        return {"ok": True, "found": False}

    def fetch(**arguments: Any) -> dict[str, Any]:
        calls.append("fetch")
        return {"ok": False, "error": "HTTP 500"}

    def comment(**arguments: Any) -> dict[str, Any]:
        calls.append("comment")
        comments.append(arguments["notes"])
        return {"ok": True}

    def unexpected(**arguments: Any) -> dict[str, Any]:
        raise AssertionError("失敗後のtoolが呼ばれました")

    result = _run(
        {
            "linkace_find_link": find,
            "fetch_web_page": fetch,
            "summarize_briefing": unexpected,
            "linkace_add_link": unexpected,
            "redmine_add_comment": comment,
        }
    )

    assert result.status == "failed"
    assert calls == ["find", "fetch"]
    assert "Webページの取得に失敗" in (result.events[0].notes or "")
    assert "HTTP 500" in (result.events[0].notes or "")


def test_scripted_skill_does_not_write_redmine_directly() -> None:
    calls: list[str] = []

    def find(**arguments: Any) -> dict[str, Any]:
        calls.append("find")
        return {"ok": True, "found": False}

    def fetch(**arguments: Any) -> dict[str, Any]:
        calls.append("fetch")
        return {
            "ok": True,
            "url": "https://example.test/article",
            "title": "Article",
            "text": "本文",
        }

    def comment(**arguments: Any) -> dict[str, Any]:
        raise AssertionError("スキルがRedmineへ直接コメントしました")

    def summarize(**arguments: Any) -> dict[str, Any]:
        calls.append("summarize")
        return {"ok": True, "briefing": "要約"}

    def add(**arguments: Any) -> dict[str, Any]:
        calls.append("add")
        return {
            "ok": True,
            "payload": {},
            "bookmark": {"action": "created", "url": "https://bookmark.test/1"},
        }

    result = _run(
        {
            "linkace_find_link": find,
            "fetch_web_page": fetch,
            "summarize_briefing": summarize,
            "linkace_add_link": add,
            "redmine_add_comment": comment,
        }
    )

    assert result.status == "processed"
    assert calls == ["find", "fetch", "summarize", "add"]


def test_scripted_skill_prefers_explicit_target_for_multiple_urls() -> None:
    requested_urls: list[str] = []

    def find(**arguments: Any) -> dict[str, Any]:
        requested_urls.append(arguments["url"])
        return {
            "ok": True,
            "found": True,
            "bookmark": {
                "has_source_list": False,
                "web_url": "https://linkace.example.test/links/12",
            },
        }

    result = _run(
        {
            "linkace_find_link": find,
            "fetch_web_page": lambda **arguments: {"ok": True},
            "summarize_briefing": lambda **arguments: {"ok": True},
            "linkace_add_link": lambda **arguments: {"ok": True},
            "redmine_add_comment": lambda **arguments: {"ok": True},
        },
        description="https://one.test と https://two.test を確認して",
        task_input={"target_url": "https://two.test"},
    )

    assert result.status == "already_done"
    assert requested_urls == ["https://two.test"]


def test_scripted_skill_rejects_ambiguous_urls_without_fetching() -> None:
    def comment(**arguments: Any) -> dict[str, Any]:
        raise AssertionError("スキルがRedmineへ直接コメントしました")

    def unexpected(**arguments: Any) -> dict[str, Any]:
        raise AssertionError("URL確定前にtoolが呼ばれました")

    result = _run(
        {
            "linkace_find_link": unexpected,
            "fetch_web_page": unexpected,
            "summarize_briefing": unexpected,
            "linkace_add_link": unexpected,
            "redmine_add_comment": comment,
        },
        description="https://one.test と https://two.test を確認して",
    )

    assert result.status == "needs_user"
    assert "一意に特定できません" in (result.events[0].notes or "")


class FakeRedmine:
    def __init__(self) -> None:
        self.updated: list[tuple[int, str]] = []

    def update_issue(self, issue_id: int, *, notes: str) -> None:
        self.updated.append((issue_id, notes))


def test_redmine_add_comment_tool_writes_and_honors_dry_run() -> None:
    root = Path(__file__).parents[1] / "tool_scripts"
    redmine = FakeRedmine()
    normal = ToolScriptCatalog(
        root,
        ToolRuntimeContext(services={"redmine_client": redmine}, settings={}),
    ).registry_for(("redmine_add_comment",))

    result = normal.execute(
        "redmine_add_comment",
        {"issue_id": 123, "notes": "進捗です。"},
        allow_writes=True,
    )

    assert result.content["ok"] is True
    assert redmine.updated == [(123, "進捗です。")]

    dry_run = ToolScriptCatalog(
        root,
        ToolRuntimeContext(
            services={"redmine_client": redmine}, settings={}, dry_run=True
        ),
    ).registry_for(("redmine_add_comment",))
    dry_result = dry_run.execute(
        "redmine_add_comment",
        {"issue_id": 123, "notes": "予定です。"},
    )

    assert dry_result.content["dry_run"] is True
    assert redmine.updated == [(123, "進捗です。")]
