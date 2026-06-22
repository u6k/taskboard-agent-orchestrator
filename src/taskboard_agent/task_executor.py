from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from taskboard_agent.llm import LLMResponse
from taskboard_agent.skill_runtime import (
    GenericSkillRunner,
    ScriptedSkillRunner,
    SkillAgentPort,
    SkillEventSink,
    SkillEvent,
    SkillExecutionResult,
    llm_response_events,
)
from taskboard_agent.skills import Skill, SkillRegistry
from taskboard_agent.tools import ToolRegistry, ToolRegistryError


PlanDecision = Literal["use_skill", "use_tools", "no_skill", "needs_user"]
GenericStatus = Literal["completed", "needs_user", "missing_tool"]
MAX_TASK_PLAN_ATTEMPTS = 3
logger = logging.getLogger(__name__)


class TaskPlanningError(RuntimeError):
    """Raised when an issue cannot be planned safely."""


class TaskLLMPort(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> LLMResponse:
        ...


class ToolCatalogPort(Protocol):
    def summaries(self) -> list[dict[str, Any]]:
        ...

    def registry_for(self, tool_names: tuple[str, ...] | list[str]) -> ToolRegistry:
        ...


@dataclass(frozen=True)
class TaskPlan:
    decision: PlanDecision
    reason: str
    skill_name: str | None = None
    tool_names: tuple[str, ...] = ()
    target_url: str | None = None
    task_input: dict[str, Any] | None = None
    user_request: str | None = None


class LiteLLMTaskPlanner:
    def __init__(self, llm: TaskLLMPort) -> None:
        self._llm = llm

    def plan(
        self,
        issue: dict[str, Any],
        skills: list[Skill],
        tools: list[dict[str, Any]],
    ) -> TaskPlan:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "あなたはRedmineチケットの作業内容を理解し、利用すべきスキルを選ぶAIです。"
                    "利用可能なtoolだけで依頼を過不足なく実行できる場合はuse_toolsにしてください。"
                    "スキルは依頼目的全体がスキルの目的と一致する場合だけ使ってください。"
                    "依頼が曖昧、必要情報が不足、または必要なtool/skillが不足している場合はneeds_userにしてください。"
                    "既存スキルなしで言語モデルだけで完了できる作業はno_skillにしてください。"
                    "出力は有効なJSON objectだけにしてください。"
                ),
            },
            {
                "role": "user",
                "content": _build_planning_prompt(issue, skills, tools),
            },
        ]
        last_error: TaskPlanningError | None = None
        for attempt in range(MAX_TASK_PLAN_ATTEMPTS):
            response = self._llm.complete(messages)
            try:
                plan = parse_task_plan(response.content)
                _validate_task_plan(plan, skills=skills, tools=tools)
                return plan
            except TaskPlanningError as exc:
                last_error = exc
                logger.warning(
                    "タスク計画の形式が不正なため修正を要求します attempt=%s/%s error=%s",
                    attempt + 1,
                    MAX_TASK_PLAN_ATTEMPTS,
                    exc,
                )
                if attempt == MAX_TASK_PLAN_ATTEMPTS - 1:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": response.content},
                        {
                            "role": "user",
                            "content": _build_plan_retry_prompt(exc),
                        },
                    ]
                )
        raise TaskPlanningError(
            f"task plan remained invalid after {MAX_TASK_PLAN_ATTEMPTS} attempts: {last_error}"
        ) from last_error


class GenericTaskRunner:
    def __init__(self, llm: TaskLLMPort) -> None:
        self._llm = llm

    def run(
        self,
        *,
        issue: dict[str, Any],
        dry_run: bool = False,
        emit_event: SkillEventSink | None = None,
    ) -> SkillExecutionResult:
        response = self._llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "あなたはRedmineチケットを処理するAIです。"
                        "利用可能な外部toolや専用skillはありません。"
                        "チケット本文だけで完了できる作業だけを実行してください。"
                        "外部システム操作、Web取得、ファイル更新、追加情報が必要な場合は完了したふりをせず、"
                        "ユーザーに求めることを返してください。出力はJSONのみです。"
                    ),
                },
                {"role": "user", "content": _build_generic_prompt(issue)},
            ]
        )
        return _parse_generic_execution(response.content, dry_run=dry_run)


class GenericToolsRunner:
    def __init__(self, skill_agent: SkillAgentPort) -> None:
        self._skill_agent = skill_agent

    def run(
        self,
        *,
        issue: dict[str, Any],
        tools: ToolRegistry,
        tool_names: tuple[str, ...],
        task_input: dict[str, Any] | None = None,
        dry_run: bool = False,
        emit_event: SkillEventSink | None = None,
    ) -> SkillExecutionResult:
        tools.require_registered(tool_names)
        llm_events: list[SkillEvent] = []

        def record_llm_response(response: LLMResponse) -> None:
            for event in llm_response_events(response):
                if emit_event is not None:
                    emit_event(event)
                else:
                    llm_events.append(event)

        agent_result = self._skill_agent.run(
            _build_tool_messages(
                issue=issue,
                tool_names=tool_names,
                task_input=task_input or {},
                dry_run=dry_run,
            ),
            tools=tools,
            allow_writes=not dry_run,
            on_llm_response=record_llm_response,
        )
        result = _parse_tool_execution(
            agent_result.final_text,
            tool_names=tool_names,
            dry_run=dry_run,
        )
        if emit_event is not None or not llm_events:
            return result
        return _insert_after_start(result, tuple(llm_events))


class TaskOrchestrator:
    def __init__(
        self,
        *,
        planner: LiteLLMTaskPlanner,
        skill_registry: SkillRegistry,
        tool_catalog: ToolCatalogPort,
        skill_agent: SkillAgentPort,
        generic_runner: GenericTaskRunner,
    ) -> None:
        self._planner = planner
        self._skill_registry = skill_registry
        self._tool_catalog = tool_catalog
        self._skill_agent = skill_agent
        self._generic_runner = generic_runner
        self._tools_runner = GenericToolsRunner(skill_agent)

    def run(
        self,
        *,
        issue: dict[str, Any],
        dry_run: bool = False,
        emit_event: SkillEventSink | None = None,
    ) -> SkillExecutionResult:
        skills = self._skill_registry.list()
        tool_summaries = self._tool_catalog.summaries()
        plan = self._planner.plan(issue, skills, tool_summaries)
        if plan.decision == "needs_user":
            return _needs_user_result(plan.user_request or plan.reason)
        if plan.decision == "no_skill":
            if emit_event is not None:
                emit_event(_plan_start_event(plan))
                return self._generic_runner.run(
                    issue=issue,
                    dry_run=dry_run,
                    emit_event=emit_event,
                )
            return _prepend_plan_event(
                self._generic_runner.run(issue=issue, dry_run=dry_run),
                _plan_notes(plan),
            )
        if plan.decision == "use_tools":
            if not plan.tool_names:
                return _needs_user_result("利用するtoolを特定できませんでした。作業内容を具体的に追記してください。")
            try:
                tool_registry = self._tool_catalog.registry_for(plan.tool_names)
            except (ToolRegistryError, RuntimeError) as exc:
                return _needs_user_result(
                    f"必要なtoolが不足しています。\n理由: {exc}"
                )
            task_input = dict(plan.task_input or {})
            if plan.target_url:
                task_input.setdefault("target_url", plan.target_url.strip())
            if emit_event is not None:
                emit_event(_plan_start_event(plan))
            result = self._tools_runner.run(
                issue=issue,
                tools=tool_registry,
                tool_names=plan.tool_names,
                task_input=task_input,
                dry_run=dry_run,
                emit_event=emit_event,
            )
            if emit_event is not None:
                return _drop_start_event(result)
            return _prepend_plan_event(result, _plan_notes(plan))

        if plan.skill_name is None:
            return _needs_user_result("利用するスキルを特定できませんでした。作業内容を具体的に追記してください。")
        skills_by_name = {skill.name: skill for skill in skills}
        skill = skills_by_name.get(plan.skill_name)
        if skill is None:
            return _needs_user_result(
                f"必要なスキル `{plan.skill_name}` が登録されていません。利用可能なスキルを追加してください。"
            )
        try:
            tool_registry = self._tool_catalog.registry_for(skill.required_tools)
        except (ToolRegistryError, RuntimeError) as exc:
            return _needs_user_result(
                f"スキル `{plan.skill_name}` に必要なtoolが不足しています。\n理由: {exc}"
            )
        if skill.runner is not None:
            runner = ScriptedSkillRunner(skill=skill, tools=tool_registry)
        else:
            runner = GenericSkillRunner(
                skill=skill,
                tools=tool_registry,
                skill_agent=self._skill_agent,
            )
        task_input = dict(plan.task_input or {})
        if plan.target_url:
            task_input.setdefault("target_url", plan.target_url.strip())
        if emit_event is not None:
            emit_event(_plan_start_event(plan))
        result = runner.run(
            issue=issue,
            task_input=task_input,
            dry_run=dry_run,
            emit_event=emit_event,
        )
        if emit_event is not None:
            return _drop_start_event(result)
        return _prepend_plan_event(result, _plan_notes(plan))


def parse_task_plan(output: str) -> TaskPlan:
    data = _loads_object(output, "task plan")
    decision = data.get("decision")
    if decision not in ("use_skill", "use_tools", "no_skill", "needs_user"):
        raise TaskPlanningError("task plan missing valid decision")
    reason = data.get("reason")
    if not isinstance(reason, str) or reason.strip() == "":
        raise TaskPlanningError("task plan missing reason")
    skill_name = _normalize_json_null(data.get("skill_name"))
    tool_names = _normalize_json_null(data.get("tool_names", []))
    target_url = _normalize_json_null(data.get("target_url"))
    task_input = _normalize_json_null(data.get("task_input"))
    user_request = _normalize_json_null(data.get("user_request"))
    if skill_name is not None and not isinstance(skill_name, str):
        raise TaskPlanningError("task plan skill_name must be a string or null")
    if target_url is not None and not isinstance(target_url, str):
        raise TaskPlanningError("task plan target_url must be a string or null")
    if task_input is not None and not isinstance(task_input, dict):
        raise TaskPlanningError("task plan task_input must be an object or null")
    if user_request is not None and not isinstance(user_request, str):
        raise TaskPlanningError("task plan user_request must be a string or null")
    if tool_names is None:
        parsed_tool_names: tuple[str, ...] = ()
    elif isinstance(tool_names, list) and all(isinstance(item, str) for item in tool_names):
        parsed_tool_names = tuple(item.strip() for item in tool_names if item.strip())
    else:
        raise TaskPlanningError("task plan tool_names must be a string list or null")
    return TaskPlan(
        decision=decision,
        reason=reason.strip(),
        skill_name=skill_name.strip() if isinstance(skill_name, str) else None,
        tool_names=parsed_tool_names,
        target_url=target_url.strip() if isinstance(target_url, str) else None,
        task_input=task_input,
        user_request=user_request.strip() if isinstance(user_request, str) else None,
    )


def _normalize_json_null(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return value


def _validate_task_plan(
    plan: TaskPlan,
    *,
    skills: list[Skill],
    tools: list[dict[str, Any]],
) -> None:
    skill_names = {skill.name for skill in skills}
    tool_names = {
        name
        for tool in tools
        if isinstance((name := tool.get("name")), str) and name
    }
    if plan.decision == "use_skill":
        if plan.skill_name is None:
            raise TaskPlanningError("use_skill requires skill_name")
        if plan.skill_name not in skill_names:
            raise TaskPlanningError(f"use_skill referenced unknown skill: {plan.skill_name}")
        if plan.tool_names:
            raise TaskPlanningError("use_skill requires tool_names to be an empty array")
        return
    if plan.decision == "use_tools":
        if plan.skill_name is not None:
            raise TaskPlanningError("use_tools requires skill_name to be null")
        if not plan.tool_names:
            raise TaskPlanningError("use_tools requires at least one tool name")
        unknown = [name for name in plan.tool_names if name not in tool_names]
        if unknown:
            raise TaskPlanningError(
                f"use_tools referenced unknown tools: {', '.join(unknown)}"
            )
        return
    if plan.skill_name is not None or plan.tool_names:
        raise TaskPlanningError(
            f"{plan.decision} requires skill_name=null and tool_names=[]"
        )
    if plan.decision == "needs_user" and not plan.user_request:
        raise TaskPlanningError("needs_user requires a non-empty user_request")


def _build_plan_retry_prompt(error: TaskPlanningError) -> str:
    return (
        "前回の出力は次の検証エラーにより無効です。\n"
        f"- {error}\n\n"
        "作業内容の判断は変えず、型と分岐の整合性だけを修正してください。\n"
        "文字列の \"null\" ではなくJSON値の null を使用してください。\n"
        "Markdownやコードフェンスを付けず、有効なJSON objectだけを返してください。"
    )


def _parse_tool_execution(
    output: str,
    *,
    tool_names: tuple[str, ...],
    dry_run: bool,
) -> SkillExecutionResult:
    data = _loads_object(output, "tool execution result")
    status = data.get("status")
    if status not in ("processed", "needs_user", "missing_tool", "failed"):
        raise TaskPlanningError("tool execution result missing valid status")
    notes = data.get("notes")
    if not isinstance(notes, str) or notes.strip() == "":
        raise TaskPlanningError("tool execution result missing notes")

    events: list[SkillEvent] = [SkillEvent("start", "作業を開始します。")]
    if status == "processed":
        events.append(SkillEvent("progress", f"toolを実行しました: {', '.join(tool_names)}"))
        events.append(SkillEvent("final_review", notes.strip()))
    elif status in ("needs_user", "missing_tool"):
        events.append(SkillEvent("final_review", notes.strip()))
    else:
        events.append(SkillEvent("final_return", notes.strip()))
    return SkillExecutionResult(
        status="dry_run" if dry_run and status == "processed" else status,
        events=tuple(events),
        target_url=_string_or_none(data.get("target_url")),
        page_title=_string_or_none(data.get("page_title")),
        briefing=_string_or_none(data.get("briefing")),
        bookmark_url=_string_or_none(data.get("bookmark_url")),
        bookmark_payload=_dict_or_none(data.get("bookmark_payload")),
        dry_run=dry_run,
    )


def _parse_generic_execution(output: str, *, dry_run: bool) -> SkillExecutionResult:
    data = _loads_object(output, "generic execution result")
    status = data.get("status")
    if status not in ("completed", "needs_user", "missing_tool"):
        raise TaskPlanningError("generic execution result missing valid status")
    notes = data.get("notes")
    if not isinstance(notes, str) or notes.strip() == "":
        raise TaskPlanningError("generic execution result missing notes")
    if status == "completed":
        return SkillExecutionResult(
            status="dry_run" if dry_run else "processed",
            events=(SkillEvent("final_review", notes.strip()),),
            dry_run=dry_run,
        )
    return SkillExecutionResult(
        status=status,
        events=(SkillEvent("final_review", notes.strip()),),
        dry_run=dry_run,
    )


def _needs_user_result(notes: str) -> SkillExecutionResult:
    return SkillExecutionResult(
        status="needs_user",
        events=(SkillEvent("final_review", notes),),
    )


def _prepend_plan_event(
    result: SkillExecutionResult,
    notes: str,
) -> SkillExecutionResult:
    events = list(result.events)
    if events and events[0].kind == "start":
        events[0] = SkillEvent("start", f"{notes}\n\n{events[0].notes}")
    else:
        events.insert(0, SkillEvent("start", f"{notes}\n\n作業を開始します。"))
    return SkillExecutionResult(
        status=result.status,
        events=tuple(events),
        target_url=result.target_url,
        page_title=result.page_title,
        briefing=result.briefing,
        bookmark_url=result.bookmark_url,
        bookmark_payload=result.bookmark_payload,
        dry_run=result.dry_run,
    )


def _plan_start_event(plan: TaskPlan) -> SkillEvent:
    return SkillEvent("start", f"{_plan_notes(plan)}\n\n作業を開始します。")


def _drop_start_event(result: SkillExecutionResult) -> SkillExecutionResult:
    events = result.events[1:] if result.events and result.events[0].kind == "start" else result.events
    return SkillExecutionResult(
        status=result.status,
        events=tuple(events),
        target_url=result.target_url,
        page_title=result.page_title,
        briefing=result.briefing,
        bookmark_url=result.bookmark_url,
        bookmark_payload=result.bookmark_payload,
        dry_run=result.dry_run,
    )


def _insert_after_start(
    result: SkillExecutionResult,
    events: tuple[SkillEvent, ...],
) -> SkillExecutionResult:
    existing = list(result.events)
    index = 1 if existing and existing[0].kind == "start" else 0
    merged = (*existing[:index], *events, *existing[index:])
    return SkillExecutionResult(
        status=result.status,
        events=tuple(merged),
        target_url=result.target_url,
        page_title=result.page_title,
        briefing=result.briefing,
        bookmark_url=result.bookmark_url,
        bookmark_payload=result.bookmark_payload,
        dry_run=result.dry_run,
    )


def _plan_notes(plan: TaskPlan) -> str:
    target = f"{plan.target_url} を対象にした作業" if plan.target_url else "チケット本文に記載された作業"
    if plan.decision == "use_skill" and plan.skill_name:
        method = f"スキル `{plan.skill_name}` を使って進めます。"
    elif plan.decision == "use_tools":
        method = f"tool `{', '.join(plan.tool_names)}` を使って進めます。"
    elif plan.decision == "no_skill":
        method = "専用スキルや外部toolは使わず、チケット本文から実行できる範囲で進めます。"
    else:
        method = "確認できた内容に沿って進めます。"
    return (
        f"ユーザーから依頼された作業は、{target}だと理解しました。"
        f"{plan.reason}と判断したためです。{method}"
    )


def _build_planning_prompt(
    issue: dict[str, Any],
    skills: list[Skill],
    tools: list[dict[str, Any]],
) -> str:
    skill_summaries = [skill.summary() for skill in skills]
    return (
        "次のチケットについて、作業内容を理解し実行方法を決めてください。\n"
        "判断時はskillとtoolの両方を検討してください。\n"
        "toolを1つまたは複数使えば依頼内容を過不足なく実行できる場合はuse_toolsを選んでください。\n"
        "toolを複数使うよりskillを使う方が依頼目的に近い場合だけuse_skillを選んでください。\n"
        "依頼されていない作業を含むスキルは選ばないでください。例えば「Webページ本文を取得して」だけなら、ブックマーク登録まで行うスキルは使わずfetch_web_pageだけを使ってください。\n"
        "型規則:\n"
        "- nullには引用符を付けず、JSON値のnullとして出力する。\n"
        "- task_inputはJSON objectまたはnull。文字列は禁止。\n"
        "- user_request、skill_name、target_urlは文字列またはnull。\n"
        "- use_skillではskill_nameを指定し、tool_namesは必ず[]にする。\n"
        "- use_toolsではskill_nameをnullにし、tool_namesを1件以上指定する。\n"
        "- no_skillではskill_nameをnull、tool_namesを[]にする。\n"
        "- needs_userではskill_nameをnull、tool_namesを[]にし、user_requestを指定する。\n"
        "- Markdownやコードフェンスを付けず、JSON objectだけを出力する。\n\n"
        "use_skillの有効な出力例:\n"
        "{\n"
        '  "decision": "use_skill",\n'
        '  "skill_name": "web-briefing-bookmark",\n'
        '  "tool_names": [],\n'
        '  "target_url": "https://example.com/article",\n'
        '  "task_input": null,\n'
        '  "reason": "依頼目的がスキルと一致するため",\n'
        '  "user_request": null\n'
        "}\n\n"
        f"利用可能なスキル:\n{json.dumps(skill_summaries, ensure_ascii=False)}\n\n"
        f"利用可能なtool:\n{json.dumps(tools, ensure_ascii=False)}\n\n"
        f"チケット:\n{json.dumps(_issue_context(issue), ensure_ascii=False)}"
    )


def _build_tool_messages(
    *,
    issue: dict[str, Any],
    tool_names: tuple[str, ...],
    task_input: dict[str, Any],
    dry_run: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "あなたはRedmineチケットを処理するAIエージェントです。"
                "指定されたfunction toolsだけを使い、依頼された範囲を過不足なく実行してください。"
                "依頼されていない追加作業は行わないでください。"
                "tool結果のok=falseは失敗として扱い、必要なら追加作業を止めてユーザー確認を求めてください。"
                "最終応答はJSONだけにしてください。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"使用するtool: {', '.join(tool_names)}\n\n"
                "実行コンテキストJSON:\n"
                f"{json.dumps({'issue': issue, 'task_input': task_input, 'dry_run': dry_run}, ensure_ascii=False)}\n\n"
                "最終応答JSON形式:\n"
                "{"
                '"status": "processed|needs_user|missing_tool|failed", '
                '"notes": "Redmineに投稿する自然な日本語の作業報告またはユーザーに求めること。実行した内容、確認したこと、成果物、未処理事項を含める。JSON文字列や内部statusの説明は書かない", '
                '"target_url": "任意", '
                '"page_title": "任意", '
                '"briefing": "任意", '
                '"bookmark_url": "任意", '
                '"bookmark_payload": "任意のobjectまたはnull"'
                "}"
            ),
        },
    ]


def _build_generic_prompt(issue: dict[str, Any]) -> str:
    return (
        "次のチケットを、外部toolや専用skillなしで実行できる範囲で処理してください。\n"
        "必ず次のJSONだけを返してください。\n"
        "{"
        '"status": "completed|needs_user|missing_tool", '
        '"notes": "Redmineに投稿する自然な日本語の作業報告またはユーザーに求めること。実行した内容、判断したこと、未処理事項を含める。JSON文字列や内部statusの説明は書かない"'
        "}\n\n"
        f"チケット:\n{json.dumps(_issue_context(issue), ensure_ascii=False)}"
    )


def _issue_context(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": issue.get("id"),
        "subject": issue.get("subject"),
        "description": issue.get("description"),
        "author": issue.get("author"),
        "assigned_to": issue.get("assigned_to"),
        "status": issue.get("status"),
        "priority": issue.get("priority"),
        "project": issue.get("project"),
        "tracker": issue.get("tracker"),
        "start_date": issue.get("start_date"),
        "due_date": issue.get("due_date"),
    }


def _loads_object(output: str, label: str) -> dict[str, Any]:
    try:
        data = json.loads(_strip_json_fence(output))
    except json.JSONDecodeError as exc:
        raise TaskPlanningError(f"{label} was not valid JSON") from exc
    if not isinstance(data, dict):
        raise TaskPlanningError(f"{label} JSON was not an object")
    return data


def _strip_json_fence(output: str) -> str:
    stripped = output.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None
