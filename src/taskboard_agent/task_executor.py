from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
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
from taskboard_agent.structured_output import (
    generic_execution_response_format,
    task_plan_response_format,
    tool_execution_response_format,
)
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
        response_format: dict[str, Any] | None = None,
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
                    "チケットの依頼内容を必ず理解した上で実行方法を判断してください。"
                    "依頼内容と整合する登録済みスキル名が明示されている場合はuse_skillにしてください。"
                    "依頼目的全体と一致するスキルがある場合もuse_skillを優先してください。"
                    "一致するスキルがなく、利用可能なtoolだけで依頼を過不足なく実行できる場合にuse_toolsにしてください。"
                    "依頼が曖昧、必要情報が不足、または必要なtool/skillが不足している場合はneeds_userにしてください。"
                    "既存スキルなしで言語モデルだけで完了できる作業はno_skillにしてください。"
                    "計画の出力構造はAPI側で指定されています。"
                ),
            },
            {
                "role": "user",
                "content": _build_planning_prompt(issue, skills, tools),
            },
        ]
        last_error: TaskPlanningError | None = None
        response_format = task_plan_response_format(
            skill_names=(skill.name for skill in skills),
            tool_names=(
                name
                for tool in tools
                if isinstance((name := tool.get("name")), str)
            ),
        )
        for attempt in range(MAX_TASK_PLAN_ATTEMPTS):
            response = self._llm.complete(
                messages,
                response_format=response_format,
            )
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
        conversation_messages: list[dict[str, Any]] | None = None,
        task_plan: TaskPlan | None = None,
    ) -> SkillExecutionResult:
        if conversation_messages:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "あなたはRedmineチケットを継続して処理するAIです。"
                        "以下の会話履歴全体と直前の再計画に従い、外部toolや専用skillなしで作業してください。"
                        "過去の成果を推測せず、会話履歴に記録された事実を使ってください。"
                        "最終応答はRedmineへ投稿する自然なMarkdown本文にしてください。"
                    ),
                },
                *conversation_messages,
                {
                    "role": "user",
                    "content": _build_conversation_generic_prompt(task_plan),
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "あなたはRedmineチケットを処理するAIです。"
                        "利用可能な外部toolや専用skillはありません。"
                        "チケット本文だけで完了できる作業だけを実行してください。"
                        "外部システム操作、Web取得、ファイル更新、追加情報が必要な場合は完了したふりをせず、"
                        "ユーザーに求めることを返してください。出力構造はAPI側で指定されています。"
                    ),
                },
                {"role": "user", "content": _build_generic_prompt(issue)},
            ]
        response = self._llm.complete(
            messages,
            response_format=(
                None if conversation_messages else generic_execution_response_format()
            ),
        )
        if conversation_messages:
            notes = response.content.strip()
            if not notes:
                raise TaskPlanningError("conversation execution result was empty")
            return SkillExecutionResult(
                status="dry_run" if dry_run else "processed",
                events=(SkillEvent("final_review", notes),),
                dry_run=dry_run,
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
            response_format=tool_execution_response_format(),
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
        plan = self.create_plan(issue)
        return self.execute_plan(
            issue=issue,
            plan=plan,
            dry_run=dry_run,
            emit_event=emit_event,
        )

    def create_plan(self, issue: dict[str, Any]) -> TaskPlan:
        return self._planner.plan(
            issue,
            self._skill_registry.list(),
            self._tool_catalog.summaries(),
        )

    def planning_catalog(self) -> tuple[list[Skill], list[dict[str, Any]]]:
        return self._skill_registry.list(), self._tool_catalog.summaries()

    def execute_plan(
        self,
        *,
        issue: dict[str, Any],
        plan: TaskPlan,
        dry_run: bool = False,
        emit_event: SkillEventSink | None = None,
        announce_plan: bool = True,
        conversation_messages: list[dict[str, Any]] | None = None,
    ) -> SkillExecutionResult:
        skills = self._skill_registry.list()
        execution_issue = dict(issue)
        if conversation_messages:
            execution_issue["conversation_context"] = conversation_messages
        if plan.decision == "needs_user":
            return _needs_user_result(plan.user_request or plan.reason)
        if plan.decision == "no_skill":
            if emit_event is not None and announce_plan:
                emit_event(_plan_start_event(plan))
            result = self._generic_runner.run(
                issue=execution_issue,
                dry_run=dry_run,
                emit_event=emit_event,
                conversation_messages=conversation_messages,
                task_plan=plan,
            )
            if emit_event is not None:
                return result
            if not announce_plan:
                return _drop_start_event(result)
            return _prepend_plan_event(result, _plan_notes(plan))
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
            if emit_event is not None and announce_plan:
                emit_event(_plan_start_event(plan))
            result = self._tools_runner.run(
                issue=execution_issue,
                tools=tool_registry,
                tool_names=plan.tool_names,
                task_input=task_input,
                dry_run=dry_run,
                emit_event=emit_event,
            )
            if emit_event is not None:
                return _drop_start_event(result)
            if not announce_plan:
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
        if emit_event is not None and announce_plan:
            emit_event(_plan_start_event(plan))
        result = runner.run(
            issue=execution_issue,
            task_input=task_input,
            dry_run=dry_run,
            emit_event=emit_event,
        )
        if emit_event is not None:
            return _drop_start_event(result)
        if not announce_plan:
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
        "APIで指定された出力構造と分岐条件に従って再回答してください。"
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
        "スキル名が明記されている場合も名前だけで機械的に選ばず、チケットの依頼内容を理解して整合性を確認してください。\n"
        "判断時はskillとtoolの両方を検討してください。\n"
        "チケットに登録済みskill名が明記され、そのskillが依頼内容と整合する場合はuse_skillで選んでください。\n"
        "依頼目的全体と一致するskillがある場合は、内部toolを個別に組み合わせずuse_skillを選んでください。\n"
        "一致するskillがなく、toolだけで依頼を過不足なく実行できる場合にuse_toolsを選んでください。\n"
        "依頼されていない作業を含むスキルは選ばないでください。例えば「Webページ本文を取得して」だけなら、ブックマーク登録まで行うスキルは使わずfetch_web_pageだけを使ってください。\n"
        "分岐規則:\n"
        "- use_skillではskill_nameを指定し、tool_namesは必ず[]にする。\n"
        "- use_toolsではskill_nameをnullにし、tool_namesを1件以上指定する。\n"
        "- no_skillではskill_nameをnull、tool_namesを[]にする。\n"
        "- needs_userではskill_nameをnull、tool_namesを[]にし、user_requestを指定する。\n"
        "task_inputは補足指示が必要な場合だけinstructionとtarget_urlを設定してください。\n"
        "出力構造と型はAPI側で指定されています。\n\n"
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
                "最終的な作業状態とRedmine向け報告を返してください。出力構造はAPI側で指定されています。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"使用するtool: {', '.join(tool_names)}\n\n"
                "実行コンテキスト:\n"
                f"{json.dumps({'issue': issue, 'task_input': task_input, 'dry_run': dry_run}, ensure_ascii=False)}\n\n"
                "notesにはRedmineへ投稿する自然な日本語の作業報告またはユーザーに求めることを記載し、"
                "実行内容、確認内容、成果物、未処理事項を含めてください。"
            ),
        },
    ]


def _build_generic_prompt(issue: dict[str, Any]) -> str:
    return (
        "次のチケットを、外部toolや専用skillなしで実行できる範囲で処理してください。\n"
        "notesにはRedmineへ投稿する自然な日本語の作業報告またはユーザーに求めることを記載し、"
        "実行内容、判断内容、未処理事項を含めてください。出力構造はAPI側で指定されています。\n\n"
        f"チケット:\n{json.dumps(_issue_context(issue), ensure_ascii=False)}"
    )


def _build_conversation_generic_prompt(plan: TaskPlan | None) -> str:
    return (
        "直前に作成した再計画を、これまでの会話コンテキストを使って実行してください。\n"
        f"再計画データ:\n{json.dumps(asdict(plan) if plan else None, ensure_ascii=False)}\n\n"
        "Redmineへ投稿する実行結果または回答の本文だけを返してください。\n"
        "会話に記録された事実と成果物を使用し、JSON、コードフェンス、前置きは付けないでください。"
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
