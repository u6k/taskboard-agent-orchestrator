from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated, Any, Protocol, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from taskboard_agent.skill_runtime import SkillEvent, SkillEventSink, SkillExecutionResult
from taskboard_agent.structured_output import revision_plan_response_format
from taskboard_agent.task_executor import (
    MAX_TASK_PLAN_ATTEMPTS,
    TaskOrchestrator,
    TaskPlan,
    TaskPlanningError,
    parse_task_plan,
)


class CheckpointerPort(Protocol):
    def get_tuple(self, config: dict[str, Any]) -> Any:
        ...


class RevisionLLMPort(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        ...


class TicketState(TypedDict, total=False):
    issue_id: int
    issue: dict[str, Any]
    initialized: bool
    dry_run: bool
    messages: Annotated[list[AnyMessage], add_messages]
    last_ingested_journal_id: int
    has_human_feedback: bool
    current_plan: dict[str, Any] | None
    completed_steps: list[str]
    artifacts: list[dict[str, Any]]
    feedback_analysis: dict[str, Any] | None
    waiting_reason: str | None
    last_result: dict[str, Any]


class RevisionPlan(TypedDict):
    previous_work_summary: str
    feedback_summary: str
    requested_changes: list[str]
    keep_existing_results: list[str]
    work_to_redo: list[str]
    task_plan: dict[str, Any]


class TicketConversationGraph:
    """Runs one durable LangGraph conversation for each Redmine issue."""

    def __init__(
        self,
        *,
        task_orchestrator: TaskOrchestrator,
        llm: RevisionLLMPort,
        checkpointer: CheckpointerPort,
        ai_user_id: int,
    ) -> None:
        self._task_orchestrator = task_orchestrator
        self._llm = llm
        self._ai_user_id = ai_user_id
        self._event_sink: SkillEventSink | None = None

        builder = StateGraph(TicketState)
        builder.add_node("initialize", self._initialize)
        builder.add_node("initial_plan", self._initial_plan)
        builder.add_node("execute_initial", self._execute_initial)
        builder.add_node("wait_for_human", self._wait_for_human)
        builder.add_node("analyze_feedback", self._analyze_feedback)
        builder.add_node("publish_revision_plan", self._publish_revision_plan)
        builder.add_node("execute_revision", self._execute_revision)
        builder.add_node("request_feedback", self._request_feedback)
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "initial_plan")
        builder.add_edge("initial_plan", "execute_initial")
        builder.add_edge("execute_initial", "wait_for_human")
        builder.add_conditional_edges(
            "wait_for_human",
            self._route_feedback,
            {True: "analyze_feedback", False: "request_feedback"},
        )
        builder.add_edge("analyze_feedback", "publish_revision_plan")
        builder.add_edge("publish_revision_plan", "execute_revision")
        builder.add_edge("execute_revision", "wait_for_human")
        builder.add_edge("request_feedback", "wait_for_human")
        self._graph = builder.compile(checkpointer=checkpointer)

    def run(
        self,
        *,
        issue: dict[str, Any],
        dry_run: bool = False,
        emit_event: SkillEventSink | None = None,
    ) -> SkillExecutionResult:
        self._event_sink = emit_event
        config = self._config(_require_issue_id(issue), dry_run=dry_run)
        try:
            snapshot = self._graph.get_state(config)
            is_interrupted = any(task.interrupts for task in snapshot.tasks)
            if snapshot.values and is_interrupted:
                cursor = int(snapshot.values.get("last_ingested_journal_id", 0))
                resume_payload = self._resume_payload(
                    issue,
                    cursor,
                    existing_messages=snapshot.values.get("messages", []),
                )
                output = self._graph.invoke(
                    Command(resume=resume_payload),
                    config=config,
                )
            elif snapshot.values and snapshot.next:
                if "execute_revision" in snapshot.next:
                    analysis = snapshot.values.get("feedback_analysis")
                    if isinstance(analysis, dict):
                        self._emit(
                            SkillEvent(
                                "start",
                                "中断した差し戻し作業を再開します。\n\n"
                                f"{_format_revision_comment(analysis)}",
                            )
                        )
                output = self._graph.invoke(None, config=config)
            else:
                output = self._graph.invoke(
                    {"issue": issue, "dry_run": dry_run}, config=config
                )
        finally:
            self._event_sink = None

        result_data = output.get("last_result")
        if not isinstance(result_data, dict):
            raise TaskPlanningError("ticket graph did not produce an execution result")
        return _execution_result_from_dict(result_data, dry_run=dry_run)

    def conversation_state(self, issue_id: int) -> dict[str, Any]:
        snapshot = self._graph.get_state(self._config(issue_id, dry_run=False))
        return dict(snapshot.values)

    @staticmethod
    def _config(issue_id: int, *, dry_run: bool) -> dict[str, Any]:
        suffix = "-dry-run" if dry_run else ""
        return {"configurable": {"thread_id": f"redmine-issue-{issue_id}{suffix}"}}

    def _initialize(self, state: TicketState) -> dict[str, Any]:
        issue = state["issue"]
        messages: list[AnyMessage] = [
            HumanMessage(content=_initial_issue_message(issue))
        ]
        for journal in _journals(issue):
            notes = _journal_notes(journal)
            if not notes:
                continue
            if _journal_user_id(journal) == self._ai_user_id:
                messages.append(AIMessage(content=notes))
            else:
                messages.append(HumanMessage(content=notes))
        return {
            "issue_id": _require_issue_id(issue),
            "issue": _without_journals(issue),
            "initialized": True,
            "dry_run": bool(state.get("dry_run", False)),
            "messages": messages,
            "last_ingested_journal_id": _max_journal_id(issue),
            "completed_steps": [],
            "artifacts": [],
            "feedback_analysis": None,
            "waiting_reason": None,
        }

    def _initial_plan(self, state: TicketState) -> dict[str, Any]:
        plan = self._task_orchestrator.create_plan(state["issue"])
        return {"current_plan": _task_plan_dict(plan)}

    def _execute_initial(self, state: TicketState) -> dict[str, Any]:
        plan = _task_plan_from_dict(state["current_plan"])
        return self._execute(state, plan=plan, announce_plan=True)

    def _wait_for_human(self, state: TicketState) -> dict[str, Any]:
        payload = interrupt(
            {
                "issue_id": state["issue_id"],
                "waiting_reason": state.get("waiting_reason"),
            }
        )
        if not isinstance(payload, dict):
            raise TaskPlanningError("ticket resume payload must be an object")
        journal_messages = payload.get("journal_messages", [])
        messages: list[AnyMessage] = []
        human_comments: list[str] = []
        for item in journal_messages:
            if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                continue
            content = item["content"].strip()
            if not content:
                continue
            if item.get("role") == "assistant":
                messages.append(AIMessage(content=content))
            else:
                human_comments.append(content)
        if human_comments:
            joined_comments = "\n\n".join(human_comments)
            messages.append(
                HumanMessage(
                    content=(
                        "このチケットは人間からAIエージェントへ差し戻されました。\n\n"
                        "人間の追加コメント:\n"
                        f"{joined_comments}\n\n"
                        "これまでの会話コンテキストを踏まえて、改めて作業を計画してください。"
                    )
                )
            )
        return {
            "issue": payload.get("issue", state["issue"]),
            "messages": messages,
            "last_ingested_journal_id": int(
                payload.get(
                    "last_ingested_journal_id",
                    state.get("last_ingested_journal_id", 0),
                )
            ),
            "has_human_feedback": bool(human_comments),
            "feedback_analysis": None,
        }

    @staticmethod
    def _route_feedback(state: TicketState) -> bool:
        return bool(state.get("has_human_feedback"))

    def _analyze_feedback(self, state: TicketState) -> dict[str, Any]:
        skills, tools = self._task_orchestrator.planning_catalog()
        revision_tools = [
            tool for tool in tools if tool.get("name") != "redmine_add_comment"
        ]
        response_text = ""
        last_error: Exception | None = None
        skill_summaries = [skill.summary() for skill in skills]
        response_format = revision_plan_response_format(
            skill_names=(skill.name for skill in skills),
            tool_names=(
                name
                for tool in revision_tools
                if isinstance((name := tool.get("name")), str)
            ),
        )
        for attempt in range(MAX_TASK_PLAN_ATTEMPTS):
            response = self._llm.complete(
                _revision_messages(
                    state,
                    skills=skill_summaries,
                    tools=revision_tools,
                    previous_response=response_text if attempt else None,
                    previous_error=str(last_error) if last_error else None,
                ),
                response_format=response_format,
            )
            response_text = response.content
            try:
                revision = _parse_revision_plan(response_text)
                plan_data = revision["task_plan"]
                plan = parse_task_plan(
                    json.dumps(
                        _normalize_revision_task_plan(plan_data),
                        ensure_ascii=False,
                    )
                )
                return {
                    "feedback_analysis": revision,
                    "current_plan": _task_plan_dict(plan),
                }
            except (TaskPlanningError, ValueError, TypeError, KeyError) as exc:
                last_error = exc
        raise TaskPlanningError(
            f"revision plan remained invalid after {MAX_TASK_PLAN_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def _publish_revision_plan(self, state: TicketState) -> dict[str, Any]:
        analysis = state.get("feedback_analysis")
        if not isinstance(analysis, dict):
            raise TaskPlanningError("revision feedback analysis is missing")
        notes = _format_revision_comment(analysis)
        self._emit(SkillEvent("start", notes))
        return {"messages": [AIMessage(content=notes)]}

    def _execute_revision(self, state: TicketState) -> dict[str, Any]:
        plan = _task_plan_from_dict(state["current_plan"])
        return self._execute(state, plan=plan, announce_plan=False)

    def _execute(
        self,
        state: TicketState,
        *,
        plan: TaskPlan,
        announce_plan: bool,
    ) -> dict[str, Any]:
        messages: list[AnyMessage] = []

        def record(event: SkillEvent) -> None:
            if event.notes:
                messages.append(AIMessage(content=event.notes))
            self._emit(event)

        result = self._task_orchestrator.execute_plan(
            issue=state["issue"],
            plan=plan,
            dry_run=bool(state.get("dry_run", False)),
            emit_event=record,
            announce_plan=announce_plan,
            conversation_messages=(
                _message_dicts(state.get("messages", []))
                if not announce_plan
                else None
            ),
        )
        completed = result.status in ("processed", "already_done", "dry_run")
        for event in result.events:
            if completed and event.kind in ("final_review", "final_return"):
                if event.notes and event.notes != "作業が終了しました。":
                    record(SkillEvent("progress", event.notes))
                continue
            record(event)
        if completed:
            record(SkillEvent("final_review", "作業が終了しました。"))
        result_artifacts = _result_artifacts(result)
        if result_artifacts:
            messages.append(
                AIMessage(
                    content=(
                        "今回の作業成果JSON:\n"
                        f"{json.dumps(result_artifacts[0], ensure_ascii=False)}"
                    )
                )
            )
        artifacts = list(state.get("artifacts", []))
        artifacts.extend(result_artifacts)
        return {
            "messages": messages,
            "last_result": _execution_result_dict(result),
            "artifacts": artifacts,
            "waiting_reason": result.status,
        }

    def _request_feedback(self, state: TicketState) -> dict[str, Any]:
        notes = "差し戻し後の修正指示を確認できませんでした。修正内容をコメントしてください。"
        event = SkillEvent("final_review", notes)
        self._emit(event)
        result = SkillExecutionResult(status="needs_user", events=())
        return {
            "messages": [AIMessage(content=notes)],
            "last_result": _execution_result_dict(result),
            "waiting_reason": "needs_user",
        }

    def _emit(self, event: SkillEvent) -> None:
        if self._event_sink is not None:
            self._event_sink(event)

    def _resume_payload(
        self,
        issue: dict[str, Any],
        cursor: int,
        *,
        existing_messages: list[AnyMessage],
    ) -> dict[str, Any]:
        journal_messages: list[dict[str, str]] = []
        existing_ai_content = {
            str(message.content).strip()
            for message in existing_messages
            if isinstance(message, AIMessage)
        }
        for journal in _journals(issue):
            journal_id = _journal_id(journal)
            notes = _journal_notes(journal)
            if not notes:
                continue
            if _journal_user_id(journal) == self._ai_user_id:
                if notes not in existing_ai_content:
                    journal_messages.append(
                        {"role": "assistant", "content": notes}
                    )
            else:
                if journal_id <= cursor:
                    continue
                journal_messages.append({"role": "user", "content": notes})
        return {
            "issue": _without_journals(issue),
            "journal_messages": journal_messages,
            "last_ingested_journal_id": max(cursor, _max_journal_id(issue)),
        }


def _revision_messages(
    state: TicketState,
    *,
    skills: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    previous_response: str | None,
    previous_error: str | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "あなたは差し戻されたRedmineチケットの再計画担当です。"
                "会話履歴全体から過去の作業と最新の人間コメントを読み取り、質問か作業依頼かにかかわらず必ず計画してください。"
                "保存済みの会話だけで回答できる場合はno_skillを選び、外部toolやskillを再実行しないでください。"
                "Redmineへの計画・結果の投稿はシステムが行います。完了済みの外部操作を理由なく繰り返してはいけません。"
                "task_inputは補足指示が必要な場合だけinstructionとtarget_urlを設定してください。"
                "出力構造と型はAPI側で指定されています。\n"
                f"利用可能なスキル: {json.dumps(skills, ensure_ascii=False)}\n"
                f"利用可能なtool: {json.dumps(tools, ensure_ascii=False)}"
            ),
        }
    ]
    state_messages = state.get("messages", [])
    last_human_index = max(
        (
            index
            for index, message in enumerate(state_messages)
            if isinstance(message, HumanMessage)
        ),
        default=-1,
    )
    for index, message in enumerate(state_messages):
        role = "assistant" if isinstance(message, AIMessage) else "user"
        content = str(message.content)
        if index == last_human_index and content.startswith(
            "Redmineへの追加コメント:\n"
        ):
            human_comment = content.removeprefix("Redmineへの追加コメント:\n")
            content = (
                "このチケットは人間からAIエージェントへ差し戻されました。\n\n"
                f"人間の追加コメント:\n{human_comment}\n\n"
                "これまでの会話コンテキストを踏まえて、改めて作業を計画してください。"
            )
        messages.append({"role": role, "content": content})
    if previous_error:
        messages.append(
            {
                "role": "system",
                "content": (
                    "前回の計画は検証条件を満たしませんでした。人間の要求として扱わず、判断内容と分岐条件の整合性を修正してください。"
                    f"\nエラー: {previous_error}\n前回の出力: {previous_response}"
                ),
            }
        )
    return messages


def _parse_revision_plan(output: str) -> RevisionPlan:
    stripped = output.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("revision plan must be an object")
    for key in ("previous_work_summary", "feedback_summary"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"revision plan missing {key}")
    for key in ("requested_changes", "keep_existing_results", "work_to_redo"):
        normalized_items = _flatten_string_list(data.get(key))
        if normalized_items is None:
            raise ValueError(f"revision plan {key} must be a string list")
        data[key] = normalized_items
    task_plan = data.get("task_plan")
    if not isinstance(task_plan, dict):
        raise ValueError("revision plan requires task_plan")
    return data  # type: ignore[return-value]


def _normalize_revision_task_plan(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    task_input = normalized.get("task_input")
    if isinstance(task_input, str) and task_input.strip():
        normalized["task_input"] = {"instruction": task_input.strip()}
    return normalized


def _flatten_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    flattened: list[str] = []
    for item in value:
        if isinstance(item, str):
            flattened.append(item)
        elif isinstance(item, list) and all(isinstance(child, str) for child in item):
            flattened.extend(item)
        else:
            return None
    return flattened


def _format_revision_comment(analysis: dict[str, Any]) -> str:
    return (
        "差し戻し内容を確認し、作業を再計画しました。\n\n"
        f"指摘された内容:\n{analysis['feedback_summary']}\n\n"
        f"今回対応する内容:\n{_bullet_text(analysis['work_to_redo'])}\n\n"
        f"維持する既存成果:\n{_bullet_text(analysis['keep_existing_results'])}\n\n"
        f"修正後の作業計画:\n{_numbered_text(analysis['work_to_redo'])}"
    )


def _bullet_text(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- なし"


def _numbered_text(items: list[str]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, 1)) or "1. 追加作業なし"


def _task_plan_dict(plan: TaskPlan) -> dict[str, Any]:
    data = asdict(plan)
    data["tool_names"] = list(plan.tool_names)
    return data


def _task_plan_from_dict(data: dict[str, Any] | None) -> TaskPlan:
    if not isinstance(data, dict):
        raise TaskPlanningError("ticket state does not contain a task plan")
    return parse_task_plan(json.dumps(data, ensure_ascii=False))


def _execution_result_dict(result: SkillExecutionResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "target_url": result.target_url,
        "page_title": result.page_title,
        "briefing": result.briefing,
        "bookmark_url": result.bookmark_url,
        "bookmark_payload": result.bookmark_payload,
        "dry_run": result.dry_run,
    }


def _execution_result_from_dict(
    data: dict[str, Any], *, dry_run: bool
) -> SkillExecutionResult:
    return SkillExecutionResult(
        status=str(data.get("status", "failed")),
        events=(),
        target_url=data.get("target_url"),
        page_title=data.get("page_title"),
        briefing=data.get("briefing"),
        bookmark_url=data.get("bookmark_url"),
        bookmark_payload=data.get("bookmark_payload"),
        dry_run=bool(data.get("dry_run", dry_run)),
    )


def _result_artifacts(result: SkillExecutionResult) -> list[dict[str, Any]]:
    artifact = {
        key: value
        for key, value in {
            "target_url": result.target_url,
            "page_title": result.page_title,
            "briefing": result.briefing,
            "bookmark_url": result.bookmark_url,
            "bookmark_payload": result.bookmark_payload,
        }.items()
        if value is not None
    }
    return [artifact] if artifact else []


def _initial_issue_message(issue: dict[str, Any]) -> str:
    return (
        f"Redmineチケット #{_require_issue_id(issue)}\n\n"
        f"件名:\n{issue.get('subject') or '(未記載)'}\n\n"
        f"依頼内容:\n{issue.get('description') or '(未記載)'}"
    )


def _message_dicts(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant" if isinstance(message, AIMessage) else "user",
            "content": str(message.content),
        }
        for message in messages
    ]


def _without_journals(issue: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in issue.items() if key != "journals"}


def _journals(issue: dict[str, Any]) -> list[dict[str, Any]]:
    journals = issue.get("journals", [])
    if not isinstance(journals, list):
        return []
    return sorted(
        (item for item in journals if isinstance(item, dict)),
        key=_journal_id,
    )


def _journal_id(journal: dict[str, Any]) -> int:
    value = journal.get("id")
    return value if isinstance(value, int) else 0


def _journal_user_id(journal: dict[str, Any]) -> int | None:
    user = journal.get("user")
    if not isinstance(user, dict):
        return None
    value = user.get("id")
    return value if isinstance(value, int) else None


def _journal_notes(journal: dict[str, Any]) -> str:
    notes = journal.get("notes")
    return notes.strip() if isinstance(notes, str) else ""


def _max_journal_id(issue: dict[str, Any]) -> int:
    return max((_journal_id(journal) for journal in _journals(issue)), default=0)


def _require_issue_id(issue: dict[str, Any]) -> int:
    issue_id = issue.get("id")
    if not isinstance(issue_id, int):
        raise TaskPlanningError("Redmine issue did not include an integer id")
    return issue_id
