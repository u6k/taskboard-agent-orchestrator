from __future__ import annotations

import json
from dataclasses import asdict, replace
from collections.abc import Collection
from typing import Any, Protocol, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

from taskboard_agent.skill_runtime import SkillEvent, SkillEventSink, SkillExecutionResult
from taskboard_agent.llm import complete_with_operation
from taskboard_agent.artifacts import ArtifactRef, ArtifactStore, InMemoryArtifactStore
from taskboard_agent.context_engine import (
    ContextEngine,
    ContextEngineError,
    ContextLimitExceeded,
    ConversationTurn,
    SessionCheckpoint,
    WorkingMemory,
)
from taskboard_agent.structured_output import revision_plan_response_format
from taskboard_agent.task_executor import (
    MAX_TASK_PLAN_ATTEMPTS,
    TaskOrchestrator,
    TaskPlan,
    TaskPlanningError,
    TaskStep,
    TaskStepExecution,
    normalize_task_plan_names,
    parse_task_plan,
    validate_task_plan,
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
    messages: list[AnyMessage]
    working_memory: dict[str, Any]
    session_checkpoint: dict[str, Any]
    recent_turns: list[dict[str, Any]]
    active_artifacts: dict[str, dict[str, Any]]
    artifact_refs: list[dict[str, Any]]
    last_ingested_journal_id: int
    has_human_feedback: bool
    current_plan: dict[str, Any] | None
    plan_steps: list[dict[str, Any]]
    current_step_index: int | None
    step_results: list[dict[str, Any]]
    run_status: str
    step_context: dict[str, Any]
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


MAX_INLINE_ASSISTANT_TURN_CHARS = 12_000


class TicketConversationGraph:
    """Runs one durable LangGraph conversation for each Redmine issue."""

    def __init__(
        self,
        *,
        task_orchestrator: TaskOrchestrator,
        llm: RevisionLLMPort,
        checkpointer: CheckpointerPort,
        ai_user_ids: Collection[int],
        context_engine: ContextEngine | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._task_orchestrator = task_orchestrator
        self._llm = llm
        self._ai_user_ids = frozenset(ai_user_ids)
        self._artifact_store = artifact_store or InMemoryArtifactStore()
        self._context_engine = context_engine or ContextEngine(
            llm=llm,
            artifact_store=self._artifact_store,
            context_window_tokens=131072,
        )
        if not self._ai_user_ids:
            raise ValueError("ai_user_ids must not be empty")
        self._event_sink: SkillEventSink | None = None

        builder = StateGraph(TicketState)
        builder.add_node("initialize", self._initialize)
        builder.add_node("initial_plan", self._initial_plan)
        builder.add_node("publish_initial_plan", self._publish_initial_plan)
        builder.add_node("select_next_step", self._select_next_step)
        builder.add_node("execute_step", self._execute_step)
        builder.add_node("finalize_execution", self._finalize_execution)
        builder.add_node("wait_for_human", self._wait_for_human)
        builder.add_node("analyze_feedback", self._analyze_feedback)
        builder.add_node("publish_revision_plan", self._publish_revision_plan)
        builder.add_node("request_feedback", self._request_feedback)
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "initial_plan")
        builder.add_edge("initial_plan", "publish_initial_plan")
        builder.add_edge("publish_initial_plan", "select_next_step")
        builder.add_conditional_edges(
            "select_next_step",
            self._route_selected_step,
            {True: "execute_step", False: "finalize_execution"},
        )
        builder.add_conditional_edges(
            "execute_step",
            self._route_after_step,
            {True: "select_next_step", False: "finalize_execution"},
        )
        builder.add_edge("finalize_execution", "wait_for_human")
        builder.add_conditional_edges(
            "wait_for_human",
            self._route_feedback,
            {True: "analyze_feedback", False: "request_feedback"},
        )
        builder.add_edge("analyze_feedback", "publish_revision_plan")
        builder.add_edge("publish_revision_plan", "select_next_step")
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
                if _is_revision_step_resume(snapshot.values, snapshot.next):
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
        turns = _initial_conversation_turns(issue, self._ai_user_ids)
        turns, initial_refs, initial_active = self._persist_initial_assistant_turns(
            turns
        )
        issue_without_journals = _without_journals(issue)
        return {
            "issue_id": _require_issue_id(issue),
            "issue": issue_without_journals,
            "initialized": True,
            "dry_run": bool(state.get("dry_run", False)),
            "messages": [],
            "working_memory": WorkingMemory(
                issue=_working_issue(issue_without_journals)
            ).to_dict(),
            "session_checkpoint": SessionCheckpoint().to_dict(),
            "recent_turns": [turn.to_dict() for turn in turns],
            "active_artifacts": initial_active,
            "artifact_refs": [ref.to_dict() for ref in initial_refs],
            "last_ingested_journal_id": _max_journal_id(issue),
            "plan_steps": [],
            "current_step_index": None,
            "step_results": [],
            "run_status": "initialized",
            "step_context": {},
            "completed_steps": [],
            "artifacts": [],
            "feedback_analysis": None,
            "waiting_reason": None,
        }

    def _initial_plan(self, state: TicketState) -> dict[str, Any]:
        try:
            prepared = self._prepare_context(state)
            artifact_ids = tuple(
                ref.artifact_id for ref in self._refs_for_state(state)
            )
            plan = _ensure_executable_steps(
                self._task_orchestrator.create_plan(
                    state["issue"],
                    context_messages=list(prepared.messages),
                    artifact_ids=artifact_ids,
                )
            )
            planned = _planned_step_state(plan)
            plan_dict = _task_plan_dict(plan)
            return {
                "current_plan": plan_dict,
                "session_checkpoint": prepared.checkpoint.to_dict(),
                "recent_turns": [turn.to_dict() for turn in prepared.recent_turns],
                "artifact_refs": [ref.to_dict() for ref in self._refs_for_state(state)],
                "artifacts": [ref.to_dict() for ref in self._refs_for_state(state)],
                "working_memory": WorkingMemory(
                    issue=_working_issue(state["issue"]),
                    current_plan=plan_dict,
                    plan_steps=tuple(planned["plan_steps"]),
                    run_status="planned",
                    active_artifacts=dict(state.get("active_artifacts", {})),
                ).to_dict(),
            } | planned
        except ContextEngineError as exc:
            plan = _context_error_plan(exc)
            planned = _planned_step_state(plan)
            plan_dict = _task_plan_dict(plan)
            return {
                "current_plan": plan_dict,
                "working_memory": WorkingMemory(
                    issue=_working_issue(state["issue"]),
                    current_plan=plan_dict,
                    plan_steps=tuple(planned["plan_steps"]),
                    run_status="planned",
                    active_artifacts=dict(state.get("active_artifacts", {})),
                ).to_dict(),
            } | planned

    def _publish_initial_plan(self, state: TicketState) -> dict[str, Any]:
        plan = _task_plan_from_dict(state["current_plan"])
        notes = f"{self._task_orchestrator.plan_notes(plan)}\n\n作業を開始します。"
        self._emit(SkillEvent("start", notes))
        return {}

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
        human_comments: list[str] = []
        human_journal_ids: list[int] = []
        for item in journal_messages:
            if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                continue
            content = item["content"].strip()
            if not content:
                continue
            if item.get("role") != "assistant":
                human_comments.append(content)
                journal_id = item.get("journal_id")
                if isinstance(journal_id, int):
                    human_journal_ids.append(journal_id)
        recent_turns = _conversation_turns(state)
        if human_comments:
            joined_comments = "\n\n".join(human_comments)
            recent_turns.append(
                ConversationTurn(
                    id=(
                        f"journal-{max(human_journal_ids)}"
                        if human_journal_ids
                        else f"resume-{len(recent_turns) + 1}"
                    ),
                    role="user",
                    content=joined_comments,
                    journal_ids=tuple(human_journal_ids),
                )
            )
        updated_issue = payload.get("issue", state["issue"])
        return {
            "issue": updated_issue,
            "recent_turns": [turn.to_dict() for turn in recent_turns],
            "working_memory": _working_memory(state, issue=updated_issue).to_dict(),
            "last_ingested_journal_id": int(
                payload.get(
                    "last_ingested_journal_id",
                    state.get("last_ingested_journal_id", 0),
                )
            ),
            "has_human_feedback": bool(human_comments),
            "feedback_analysis": None,
            "messages": [],
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
            artifact_ids=(ref.artifact_id for ref in self._refs_for_state(state)),
        )
        try:
            prepared = self._prepare_context(state)
        except ContextEngineError as exc:
            plan = _context_error_plan(exc)
            planned = _planned_step_state(plan)
            plan_dict = _task_plan_dict(plan)
            analysis = {
                "previous_work_summary": "context組み立てに失敗しました。",
                "feedback_summary": str(exc),
                "requested_changes": [],
                "keep_existing_results": [],
                "work_to_redo": [],
                "task_plan": plan_dict,
            }
            return {
                "feedback_analysis": analysis,
                "current_plan": plan_dict,
                "working_memory": WorkingMemory(
                    issue=_working_issue(state["issue"]),
                    current_plan=plan_dict,
                    plan_steps=tuple(planned["plan_steps"]),
                    run_status="planned",
                    active_artifacts=dict(state.get("active_artifacts", {})),
                ).to_dict(),
            } | planned
        for attempt in range(MAX_TASK_PLAN_ATTEMPTS):
            response = complete_with_operation(
                self._llm,
                _revision_messages(
                    context_messages=list(prepared.messages),
                    skills=skill_summaries,
                    tools=revision_tools,
                    previous_response=response_text if attempt else None,
                    previous_error=str(last_error) if last_error else None,
                ),
                response_format=response_format,
                operation="revision_plan",
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
                plan = normalize_task_plan_names(plan, skills=skills, tools=revision_tools)
                validate_task_plan(
                    plan,
                    skills=skills,
                    tools=revision_tools,
                    artifact_ids={ref.artifact_id for ref in self._refs_for_state(state)},
                )
                plan = _ensure_executable_steps(plan)
                normalized_revision = dict(revision)
                plan_dict = _task_plan_dict(plan)
                planned = _planned_step_state(plan)
                normalized_revision["task_plan"] = plan_dict
                return {
                    "feedback_analysis": normalized_revision,
                    "current_plan": plan_dict,
                    "session_checkpoint": prepared.checkpoint.to_dict(),
                    "recent_turns": [
                        turn.to_dict() for turn in prepared.recent_turns
                    ],
                    "artifact_refs": [
                        ref.to_dict() for ref in self._refs_for_state(state)
                    ],
                    "artifacts": [
                        ref.to_dict() for ref in self._refs_for_state(state)
                    ],
                    "working_memory": WorkingMemory(
                        issue=_working_issue(state["issue"]),
                        current_plan=plan_dict,
                        plan_steps=tuple(planned["plan_steps"]),
                        run_status="planned",
                        active_artifacts=dict(state.get("active_artifacts", {})),
                    ).to_dict(),
                } | planned
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
        return {}

    def _select_next_step(self, state: TicketState) -> dict[str, Any]:
        plan_steps = list(state.get("plan_steps", []))
        for index, step in enumerate(plan_steps):
            if step.get("status") in ("pending", "running"):
                updated_step = dict(step)
                updated_step["status"] = "running"
                plan_steps[index] = updated_step
                started_event = _step_started_event(updated_step)
                self._emit(started_event)
                return {
                    "plan_steps": plan_steps,
                    "current_step_index": index,
                    "run_status": "running",
                }
        return {"current_step_index": None}

    @staticmethod
    def _route_selected_step(state: TicketState) -> bool:
        return state.get("current_step_index") is not None

    def _execute_step(self, state: TicketState) -> dict[str, Any]:
        plan = _task_plan_from_dict(state["current_plan"])
        current_step_index = state.get("current_step_index")
        if current_step_index is None:
            raise TaskPlanningError("current step index is missing")
        if current_step_index < 0 or current_step_index >= len(plan.steps):
            raise TaskPlanningError("current step index is out of range")

        issue = dict(state["issue"])
        step = plan.steps[current_step_index]
        try:
            selected_artifact_ids = _selected_step_artifact_ids(
                state, step=step, step_index=current_step_index
            )
            prepared = self._prepare_context(
                state, selected_artifact_ids=selected_artifact_ids
            )
            issue["selected_artifacts"] = [
                {
                    "artifact_id": artifact_id,
                    "content": self._artifact_store.get(artifact_id),
                }
                for artifact_id in selected_artifact_ids
            ]
            execution = self._task_orchestrator.execute_single_step(
                issue=issue,
                plan=plan,
                step=step,
                step_index=current_step_index + 1,
                dry_run=bool(state.get("dry_run", False)),
                step_context=list(prepared.messages),
            )
        except ContextEngineError as exc:
            result = SkillExecutionResult(
                status="needs_user",
                events=(SkillEvent("final_review", _context_error_notes(exc)),),
                assistant_turn_text=_context_error_notes(exc),
                dry_run=bool(state.get("dry_run", False)),
            )
            execution = TaskStepExecution(
                index=current_step_index + 1,
                step=step,
                result=result,
                events=result.events,
                terminal_status="needs_user",
            )
            prepared = None

        result_refs, active_artifacts = self._persist_step_artifacts(
            state,
            execution.result,
            step=step,
            step_index=current_step_index,
        )
        execution_result = replace(
            execution.result,
            artifacts=tuple(ref.to_dict() for ref in result_refs),
        )
        step_context = dict(state.get("step_context", {}))
        step_context["last_step_status"] = execution.result.status
        if execution.terminal_status:
            step_context["terminal_status"] = execution.terminal_status

        for event in execution.events:
            recorded = _step_event_for_redmine(
                event,
                terminal=execution.should_stop,
            )
            self._emit(recorded)
        artifacts = list(state.get("artifacts", []))
        artifacts.extend(ref.to_dict() for ref in result_refs)

        plan_steps = _record_executed_step(
            state.get("plan_steps", []),
            index=current_step_index,
            result=execution_result,
            artifacts=[ref.to_dict() for ref in result_refs],
        )
        step_status_event = _step_status_event(
            step=plan_steps[current_step_index],
            status=execution.result.status,
        )
        self._emit(step_status_event)
        step_results = list(state.get("step_results", []))
        step_results.append(
            {
                "step_id": plan_steps[current_step_index].get("id"),
                "index": current_step_index,
                "status": execution_result.status,
                "result": _execution_result_dict(execution_result),
                "events": [_skill_event_dict(event) for event in execution.events],
                "artifacts": [ref.to_dict() for ref in result_refs],
            }
        )
        run_status = execution.result.status
        return {
            "artifacts": artifacts,
            "artifact_refs": [ref.to_dict() for ref in _artifact_refs(state)]
            + [ref.to_dict() for ref in result_refs],
            "active_artifacts": active_artifacts,
            "plan_steps": plan_steps,
            "current_step_index": (
                current_step_index if execution.should_stop else None
            ),
            "step_results": step_results,
            "run_status": run_status,
            "step_context": step_context,
            "working_memory": WorkingMemory(
                issue=_working_issue(issue),
                current_plan=_task_plan_dict(plan),
                plan_steps=tuple(plan_steps),
                run_status=run_status,
                waiting_reason=(run_status if execution.should_stop else None),
                active_artifacts=active_artifacts,
            ).to_dict(),
            **(
                {
                    "session_checkpoint": prepared.checkpoint.to_dict(),
                    "recent_turns": [
                        turn.to_dict() for turn in prepared.recent_turns
                    ],
                }
                if prepared is not None
                else {}
            ),
        }

    @staticmethod
    def _route_after_step(state: TicketState) -> bool:
        if _current_step_is_terminal(state):
            return False
        return _next_pending_step_index(state.get("plan_steps", [])) is not None

    def _prepare_context(
        self,
        state: TicketState,
        *,
        selected_artifact_ids: tuple[str, ...] = (),
    ) -> Any:
        refs = self._refs_for_state(state)
        return self._context_engine.prepare(
            working_memory=_working_memory(state),
            checkpoint=SessionCheckpoint.from_dict(state.get("session_checkpoint")),
            recent_turns=self._turns_for_state(state),
            artifact_refs=refs,
            selected_artifact_ids=selected_artifact_ids,
        )

    def _refs_for_state(self, state: TicketState) -> list[ArtifactRef]:
        refs = _artifact_refs(state)
        known = {ref.artifact_id for ref in refs}
        for index, value in enumerate(state.get("artifacts", []), 1):
            if not isinstance(value, dict) or "artifact_id" in value:
                continue
            ref = self._artifact_store.put(
                value,
                kind="legacy_artifact",
                source_step_id=f"legacy-{index}",
                label=f"legacy-artifact-{index}",
            )
            if ref.artifact_id not in known:
                refs.append(ref)
                known.add(ref.artifact_id)
        for turn in _conversation_turns(state):
            if turn.role != "assistant" or len(turn.content) <= MAX_INLINE_ASSISTANT_TURN_CHARS:
                continue
            ref = self._artifact_store.put(
                {"text": turn.content},
                kind="assistant_turn",
                source_turn_id=turn.id,
                label="legacy-assistant-answer",
            )
            if ref.artifact_id not in known:
                refs.append(ref)
                known.add(ref.artifact_id)
        return refs

    def _turns_for_state(self, state: TicketState) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        for turn in _conversation_turns(state):
            if turn.role != "assistant" or len(turn.content) <= MAX_INLINE_ASSISTANT_TURN_CHARS:
                turns.append(turn)
                continue
            ref = self._artifact_store.put(
                {"text": turn.content},
                kind="assistant_turn",
                source_turn_id=turn.id,
                label="legacy-assistant-answer",
            )
            turns.append(
                replace(
                    turn,
                    content=(
                        "長文のlegacy assistant回答は成果物へ保存されました。\n"
                        f"artifact_id: {ref.artifact_id}\nlabel: legacy-assistant-answer"
                    ),
                )
            )
        return turns

    def _persist_step_artifacts(
        self,
        state: TicketState,
        result: SkillExecutionResult,
        *,
        step: TaskStep,
        step_index: int,
    ) -> tuple[list[ArtifactRef], dict[str, dict[str, Any]]]:
        source_turn_id = _latest_turn_id(state)
        source_step_id = f"step-{step_index + 1}"
        refs: list[ArtifactRef] = []
        for artifact_index, artifact in enumerate(result.artifacts, 1):
            refs.append(
                self._artifact_store.put(
                    artifact,
                    kind="tool_output" if step.kind == "tool" else "step_artifact",
                    source_turn_id=source_turn_id,
                    source_step_id=source_step_id,
                    label=f"{source_step_id}-artifact-{artifact_index}",
                )
            )
        step_output = {
            "status": result.status,
            "assistant_turn_text": _canonical_result_text(result),
            "target_url": result.target_url,
            "page_title": result.page_title,
            "briefing": result.briefing,
            "bookmark_url": result.bookmark_url,
            "bookmark_payload": result.bookmark_payload,
        }
        refs.append(
            self._artifact_store.put(
                step_output,
                kind="step_output",
                source_turn_id=source_turn_id,
                source_step_id=source_step_id,
                label=step.output_artifact_name or source_step_id,
            )
        )
        refs = _deduplicate_refs(refs)
        active = {
            key: dict(value)
            for key, value in state.get("active_artifacts", {}).items()
            if isinstance(value, dict)
        }
        logical_name = step.output_artifact_name or source_step_id
        previous = active.get(logical_name, {})
        previous_version = previous.get("version")
        version = previous_version + 1 if isinstance(previous_version, int) else 1
        active[logical_name] = {
            "version": version,
            "artifact_id": refs[-1].artifact_id,
        }
        return refs, active

    def _finalize_execution(self, state: TicketState) -> dict[str, Any]:
        plan = _task_plan_from_dict(state["current_plan"])
        if not state.get("plan_steps") and plan.decision == "needs_user":
            answer = plan.user_request or plan.reason
            result = SkillExecutionResult(
                status="needs_user",
                events=(SkillEvent("final_review", answer),),
                assistant_turn_text=answer,
                dry_run=bool(state.get("dry_run", False)),
            )
            event = result.events[0]
            self._emit(event)
            turn_update = self._record_assistant_turn(state, answer)
            result = replace(
                result,
                artifacts=tuple(turn_update["assistant_artifact_refs"]),
            )
            return {
                "last_result": _execution_result_dict(
                    result, include_assistant_turn=True
                ),
                "waiting_reason": result.status,
                "run_status": result.status,
                "recent_turns": turn_update["recent_turns"],
                "artifact_refs": turn_update["artifact_refs"],
                "artifacts": turn_update["artifact_refs"],
                "active_artifacts": turn_update["active_artifacts"],
                "working_memory": WorkingMemory(
                    issue=_working_issue(state["issue"]),
                    current_plan=_task_plan_dict(plan),
                    plan_steps=tuple(state.get("plan_steps", [])),
                    run_status=result.status,
                    waiting_reason=result.status,
                    active_artifacts=turn_update["active_artifacts"],
                ).to_dict(),
            }

        status = _final_execution_status(
            state,
            dry_run=bool(state.get("dry_run", False)),
        )
        final_step_event = self._task_orchestrator.step_final_event(
            plan=plan,
            status=status,
        )
        if status in ("processed", "already_done", "dry_run"):
            progress_event = SkillEvent("progress", final_step_event.notes)
            self._emit(progress_event)
            review_event = SkillEvent("final_review", "作業が終了しました。")
            self._emit(review_event)
        else:
            self._emit(final_step_event)

        answer = self._latest_step_assistant_text(state) or final_step_event.notes or "作業が終了しました。"
        turn_update = self._record_assistant_turn(state, answer)

        result = SkillExecutionResult(
            status=status,
            events=(),
            artifacts=tuple(turn_update["assistant_artifact_refs"]),
            assistant_turn_text=answer,
            dry_run=bool(state.get("dry_run", False)),
        )
        step_context = dict(state.get("step_context", {}))
        step_context["last_result_status"] = status
        return {
            "last_result": _execution_result_dict(
                result, include_assistant_turn=True
            ),
            "current_step_index": None if status in ("processed", "already_done", "dry_run") else state.get("current_step_index"),
            "run_status": status,
            "step_context": step_context,
            "waiting_reason": status,
            "recent_turns": turn_update["recent_turns"],
            "artifact_refs": turn_update["artifact_refs"],
            "artifacts": turn_update["artifact_refs"],
            "active_artifacts": turn_update["active_artifacts"],
            "working_memory": WorkingMemory(
                issue=_working_issue(state["issue"]),
                current_plan=_task_plan_dict(plan),
                plan_steps=tuple(state.get("plan_steps", [])),
                run_status=status,
                waiting_reason=status,
                active_artifacts=turn_update["active_artifacts"],
            ).to_dict(),
        }

    def _request_feedback(self, state: TicketState) -> dict[str, Any]:
        notes = "差し戻し後の修正指示を確認できませんでした。修正内容をコメントしてください。"
        event = SkillEvent("final_review", notes)
        self._emit(event)
        result = SkillExecutionResult(status="needs_user", events=())
        turn_update = self._record_assistant_turn(state, notes)
        result = replace(
            result,
            assistant_turn_text=notes,
            artifacts=tuple(turn_update["assistant_artifact_refs"]),
        )
        return {
            "last_result": _execution_result_dict(
                result, include_assistant_turn=True
            ),
            "waiting_reason": "needs_user",
            "run_status": "needs_user",
            "recent_turns": turn_update["recent_turns"],
            "artifact_refs": turn_update["artifact_refs"],
            "artifacts": turn_update["artifact_refs"],
            "active_artifacts": turn_update["active_artifacts"],
            "working_memory": WorkingMemory(
                issue=_working_issue(state["issue"]),
                current_plan=(
                    dict(state["current_plan"])
                    if isinstance(state.get("current_plan"), dict)
                    else None
                ),
                plan_steps=tuple(state.get("plan_steps", [])),
                run_status="needs_user",
                waiting_reason="needs_user",
                active_artifacts=turn_update["active_artifacts"],
            ).to_dict(),
        }

    def _record_assistant_turn(
        self, state: TicketState, text: str
    ) -> dict[str, Any]:
        recent_turns = _conversation_turns(state)
        turn_id = f"assistant-{len(recent_turns) + 1}"
        ref = self._artifact_store.put(
            {"text": text},
            kind="assistant_turn",
            source_turn_id=turn_id,
            label="assistant-answer",
        )
        turn_content = (
            text
            if len(text) <= MAX_INLINE_ASSISTANT_TURN_CHARS
            else (
                "長文のassistant回答は成果物へ保存されました。\n"
                f"artifact_id: {ref.artifact_id}\nlabel: assistant-answer"
            )
        )
        turn = ConversationTurn(id=turn_id, role="assistant", content=turn_content)
        recent_turns.append(turn)
        refs = _deduplicate_refs([*self._refs_for_state(state), ref])
        active = {
            key: dict(value)
            for key, value in state.get("active_artifacts", {}).items()
            if isinstance(value, dict)
        }
        previous = active.get("assistant-answer", {})
        previous_version = previous.get("version")
        active["assistant-answer"] = {
            "version": previous_version + 1 if isinstance(previous_version, int) else 1,
            "artifact_id": ref.artifact_id,
        }
        return {
            "recent_turns": [item.to_dict() for item in recent_turns],
            "artifact_refs": [item.to_dict() for item in refs],
            "assistant_artifact_refs": [ref.to_dict()],
            "active_artifacts": active,
        }

    def _persist_initial_assistant_turns(
        self, turns: list[ConversationTurn]
    ) -> tuple[
        list[ConversationTurn],
        list[ArtifactRef],
        dict[str, dict[str, Any]],
    ]:
        persisted: list[ConversationTurn] = []
        refs: list[ArtifactRef] = []
        active: dict[str, dict[str, Any]] = {}
        version = 0
        for turn in turns:
            if turn.role != "assistant":
                persisted.append(turn)
                continue
            ref = self._artifact_store.put(
                {"text": turn.content},
                kind="assistant_turn",
                source_turn_id=turn.id,
                label="assistant-answer",
            )
            refs.append(ref)
            version += 1
            active["assistant-answer"] = {
                "version": version,
                "artifact_id": ref.artifact_id,
            }
            if len(turn.content) > MAX_INLINE_ASSISTANT_TURN_CHARS:
                persisted.append(
                    replace(
                        turn,
                        content=(
                            "長文のassistant回答は成果物へ保存されました。\n"
                            f"artifact_id: {ref.artifact_id}\nlabel: assistant-answer"
                        ),
                    )
                )
            else:
                persisted.append(turn)
        return persisted, _deduplicate_refs(refs), active

    def _latest_step_assistant_text(self, state: TicketState) -> str | None:
        for step_result in reversed(state.get("step_results", [])):
            artifacts = step_result.get("artifacts") if isinstance(step_result, dict) else None
            if not isinstance(artifacts, list):
                continue
            for artifact in reversed(artifacts):
                if not isinstance(artifact, dict) or artifact.get("kind") != "step_output":
                    continue
                artifact_id = artifact.get("artifact_id")
                if not isinstance(artifact_id, str):
                    continue
                content = self._artifact_store.get(artifact_id)
                text = content.get("assistant_turn_text") if isinstance(content, dict) else None
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return None

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
        journal_messages: list[dict[str, Any]] = []
        for journal in _journals(issue):
            journal_id = _journal_id(journal)
            notes = _journal_notes(journal)
            if not notes:
                continue
            if _journal_user_id(journal) in self._ai_user_ids:
                continue
            else:
                if journal_id <= cursor:
                    continue
                journal_messages.append(
                    {"role": "user", "content": notes, "journal_id": journal_id}
                )
        return {
            "issue": _without_journals(issue),
            "journal_messages": journal_messages,
            "last_ingested_journal_id": max(cursor, _max_journal_id(issue)),
        }


def _revision_messages(
    *,
    context_messages: list[dict[str, Any]],
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
                "session checkpoint、直近会話、artifact catalogから過去の作業と最新の人間コメントを読み取り、質問か作業依頼かにかかわらず必ず計画してください。"
                "保存済みの会話だけで回答できる場合はno_skillを選び、外部toolやskillを再実行しないでください。"
                "Redmineへの計画・結果の投稿はシステムが行います。完了済みの外部操作を理由なく繰り返してはいけません。"
                "再計画では必ず作業ステップに分解し、各ステップをskill/tool/llm/unavailableのいずれかに分類してください。"
                "専用toolやskillがない作業でも、LLMで可能な要約・分析・比較・提案・文章作成はllmステップとして計画し、"
                "実行できないことだけunavailableステップまたはlimitationsに明示してください。"
                "toolステップのnameには利用可能なtoolのnameだけを正確に入れてください。説明文、表示名、括弧付き表記を混ぜないでください。"
                "ユーザーが `web_search_pages` のようなtool名を明示した場合、nameにはその文字列だけを入れてください。"
                "skillステップのnameにも利用可能なskillのnameだけを正確に入れてください。"
                "depends_onは同一計画内の先行step番号だけを1始まりで指定してください。"
                "input_artifact_idsはartifact catalogに存在し、そのstepで本文を読む成果物だけを指定してください。"
                "output_artifact_nameは生成成果物の安定した論理名を指定し、不要ならnullにしてください。"
                "過去成果物への参照候補が複数あり一意に選べない場合は、推測せずneeds_userで確認してください。"
                "task_inputは補足指示が必要な場合だけinstructionとtarget_urlを設定してください。"
                "出力構造と型はAPI側で指定されています。\n"
                f"利用可能なスキル: {json.dumps(skills, ensure_ascii=False)}\n"
                f"利用可能なtool: {json.dumps(tools, ensure_ascii=False)}"
            ),
        }
    ]
    messages.extend(context_messages)
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
    task_plan = analysis.get("task_plan")
    step_text = _revision_step_text(task_plan if isinstance(task_plan, dict) else {})
    limitation_text = _revision_limitation_text(task_plan if isinstance(task_plan, dict) else {})
    return (
        "差し戻し内容を確認し、作業を再計画しました。\n\n"
        f"指摘された内容:\n{analysis['feedback_summary']}\n\n"
        f"今回対応する内容:\n{_bullet_text(analysis['work_to_redo'])}\n\n"
        f"維持する既存成果:\n{_bullet_text(analysis['keep_existing_results'])}\n\n"
        f"修正後の作業計画:\n{step_text or _numbered_text(analysis['work_to_redo'])}"
        f"{limitation_text}"
    )


def _revision_step_text(task_plan: dict[str, Any]) -> str:
    steps = task_plan.get("steps")
    if not isinstance(steps, list):
        return ""
    lines: list[str] = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        kind = step.get("kind") or "unknown"
        purpose = step.get("purpose") or "(目的未記載)"
        lines.append(f"{index}. {kind}: {purpose}")
    return "\n".join(lines)


def _revision_limitation_text(task_plan: dict[str, Any]) -> str:
    limitations = task_plan.get("limitations")
    if not isinstance(limitations, list):
        return ""
    items = [item for item in limitations if isinstance(item, str) and item.strip()]
    if not items:
        return ""
    return "\n\n実行できないこと・未確認事項:\n" + "\n".join(f"- {item}" for item in items)


def _bullet_text(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- なし"


def _numbered_text(items: list[str]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, 1)) or "1. 追加作業なし"


def _task_plan_dict(plan: TaskPlan) -> dict[str, Any]:
    data = asdict(plan)
    data["tool_names"] = list(plan.tool_names)
    data["steps"] = [asdict(step) for step in plan.steps]
    data["limitations"] = list(plan.limitations)
    return data


def _planned_step_state(plan: TaskPlan) -> dict[str, Any]:
    plan_steps = [
        _plan_step_state(index=index, step=step)
        for index, step in enumerate(plan.steps)
    ]
    return {
        "plan_steps": plan_steps,
        "current_step_index": 0 if plan_steps else None,
        "step_results": [],
        "run_status": "planned",
        "step_context": {
            "decision": plan.decision,
            "reason": plan.reason,
            "limitations": list(plan.limitations),
            "execution_model": "langgraph_step_loop",
        },
    }


def _ensure_executable_steps(plan: TaskPlan) -> TaskPlan:
    if plan.steps or plan.decision == "needs_user":
        return plan
    if plan.decision == "no_skill":
        instruction = None
        if isinstance(plan.task_input, dict):
            value = plan.task_input.get("instruction")
            instruction = value if isinstance(value, str) and value.strip() else None
        return replace(
            plan,
            steps=(
                TaskStep(
                    kind="llm",
                    purpose=instruction or plan.reason,
                    arguments=plan.task_input,
                ),
            ),
        )
    if plan.decision == "use_skill":
        if not plan.skill_name:
            return replace(
                plan,
                decision="needs_user",
                user_request="利用するスキルを特定できませんでした。作業内容を具体的に追記してください。",
            )
        return replace(
            plan,
            steps=(
                TaskStep(
                    kind="skill",
                    name=plan.skill_name,
                    purpose=plan.reason,
                    arguments=plan.task_input,
                ),
            ),
        )
    if not plan.tool_names:
        return replace(
            plan,
            decision="needs_user",
            user_request="利用するtoolを特定できませんでした。作業内容を具体的に追記してください。",
        )
    return replace(
        plan,
        steps=tuple(
            TaskStep(
                kind="tool",
                name=tool_name,
                purpose=plan.reason,
                arguments=plan.task_input,
            )
            for tool_name in plan.tool_names
        ),
    )


def _plan_step_state(*, index: int, step: Any) -> dict[str, Any]:
    return {
        "id": f"step-{index + 1}",
        "index": index,
        "kind": step.kind,
        "name": step.name,
        "purpose": step.purpose,
        "arguments": step.arguments,
        "depends_on": list(step.depends_on),
        "input_artifact_ids": list(step.input_artifact_ids),
        "output_artifact_name": step.output_artifact_name,
        "status": "pending",
        "result": None,
        "error": None,
        "artifacts": [],
    }


def _record_executed_step(
    plan_steps: list[dict[str, Any]],
    *,
    index: int,
    result: SkillExecutionResult,
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    updated_steps = [dict(step) for step in plan_steps]
    if index < 0 or index >= len(updated_steps):
        raise TaskPlanningError("executed step index is out of range")
    step = updated_steps[index]
    step["status"] = _step_status_from_result(result)
    step["result"] = _execution_result_dict(result)
    step["error"] = _event_notes(result.events) if step["status"] == "failed" else None
    step["artifacts"] = artifacts
    return updated_steps


def _step_status_from_result(result: SkillExecutionResult) -> str:
    if result.status == "skipped":
        return "skipped"
    if result.status in ("processed", "already_done", "dry_run"):
        return "completed"
    if result.status in ("needs_user", "missing_tool"):
        return "needs_user"
    return "failed"


def _skill_event_dict(event: SkillEvent) -> dict[str, Any]:
    return {"kind": event.kind, "notes": event.notes}


def _event_notes(events: tuple[SkillEvent, ...]) -> str:
    notes = [event.notes for event in events if event.notes]
    return "\n\n".join(notes) if notes else ""


def _step_started_event(step: dict[str, Any]) -> SkillEvent:
    return SkillEvent("progress", f"{_step_label(step)} を開始しました: {_step_purpose(step)}")


def _step_status_event(*, step: dict[str, Any], status: str) -> SkillEvent:
    label = _step_label(step)
    purpose = _step_purpose(step)
    if status in ("processed", "already_done", "dry_run"):
        verb = "完了しました"
    elif status == "skipped":
        verb = "スキップしました"
    elif status in ("needs_user", "missing_tool"):
        verb = "判断待ちで停止しました"
    else:
        verb = "失敗しました"
    return SkillEvent("progress", f"{label} を{verb}: {purpose}")


def _step_label(step: dict[str, Any]) -> str:
    index = step.get("index")
    if isinstance(index, int):
        return f"ステップ {index + 1}"
    return "ステップ"


def _step_purpose(step: dict[str, Any]) -> str:
    purpose = step.get("purpose")
    return purpose if isinstance(purpose, str) and purpose.strip() else "(目的未記載)"


def _step_event_for_redmine(event: SkillEvent, *, terminal: bool) -> SkillEvent:
    if not terminal and event.kind in ("final_review", "final_return"):
        return SkillEvent("progress", event.notes)
    return event


def _next_pending_step_index(plan_steps: list[dict[str, Any]]) -> int | None:
    for index, step in enumerate(plan_steps):
        if step.get("status") in ("pending", "running"):
            return index
    return None


def _current_step_is_terminal(state: TicketState) -> bool:
    current_step_index = state.get("current_step_index")
    if current_step_index is None:
        return False
    plan_steps = state.get("plan_steps", [])
    if current_step_index < 0 or current_step_index >= len(plan_steps):
        return True
    return plan_steps[current_step_index].get("status") in (
        "failed",
        "needs_user",
    )


def _final_execution_status(state: TicketState, *, dry_run: bool) -> str:
    for step in state.get("plan_steps", []):
        if step.get("status") == "failed":
            return "failed"
        if step.get("status") == "needs_user":
            result = step.get("result")
            if isinstance(result, dict) and result.get("status") == "missing_tool":
                return "missing_tool"
            return "needs_user"
    return "dry_run" if dry_run else "processed"


def _is_revision_step_resume(
    values: dict[str, Any],
    next_nodes: tuple[str, ...],
) -> bool:
    if not isinstance(values.get("feedback_analysis"), dict):
        return False
    return any(
        node in {"select_next_step", "execute_step", "finalize_execution"}
        for node in next_nodes
    )


def _collapsed_execution_step_state(
    state: TicketState,
    result_status: str,
) -> dict[str, Any]:
    plan_steps = list(state.get("plan_steps", []))
    if result_status in ("processed", "already_done", "dry_run"):
        plan_steps = [_completed_collapsed_step(step) for step in plan_steps]
        current_step_index = None
    else:
        current_step_index = state.get("current_step_index")
    step_context = dict(state.get("step_context", {}))
    step_context["last_result_status"] = result_status
    return {
        "plan_steps": plan_steps,
        "current_step_index": current_step_index,
        "run_status": result_status,
        "step_context": step_context,
    }


def _completed_collapsed_step(step: dict[str, Any]) -> dict[str, Any]:
    updated = dict(step)
    updated["status"] = (
        "skipped" if updated.get("kind") == "unavailable" else "completed"
    )
    return updated


def _task_plan_from_dict(data: dict[str, Any] | None) -> TaskPlan:
    if not isinstance(data, dict):
        raise TaskPlanningError("ticket state does not contain a task plan")
    return parse_task_plan(json.dumps(data, ensure_ascii=False))


def _execution_result_dict(
    result: SkillExecutionResult, *, include_assistant_turn: bool = False
) -> dict[str, Any]:
    data = {
        "status": result.status,
        "target_url": result.target_url,
        "page_title": result.page_title,
        "briefing": None,
        "bookmark_url": result.bookmark_url,
        "bookmark_payload": None,
        "artifacts": list(result.artifacts),
        "dry_run": result.dry_run,
    }
    if include_assistant_turn:
        data["assistant_turn_text"] = result.assistant_turn_text
    return data


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
        artifacts=tuple(
            item for item in data.get("artifacts", []) if isinstance(item, dict)
        ),
        assistant_turn_text=(
            str(data["assistant_turn_text"])
            if data.get("assistant_turn_text") is not None
            else None
        ),
        dry_run=bool(data.get("dry_run", dry_run)),
    )


def _result_artifacts(result: SkillExecutionResult) -> list[dict[str, Any]]:
    artifacts = list(result.artifacts)
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
    if artifact:
        artifacts.append(artifact)
    return artifacts


def _initial_issue_message(issue: dict[str, Any]) -> str:
    return (
        f"Redmineチケット #{_require_issue_id(issue)}\n\n"
        f"件名:\n{issue.get('subject') or '(未記載)'}\n\n"
        f"依頼内容:\n{issue.get('description') or '(未記載)'}"
    )


def _initial_conversation_turns(
    issue: dict[str, Any], ai_user_ids: Collection[int]
) -> list[ConversationTurn]:
    turns = [
        ConversationTurn(
            id=f"issue-{_require_issue_id(issue)}",
            role="user",
            content=_initial_issue_message(issue),
        )
    ]
    pending_human: list[str] = []
    pending_ids: list[int] = []

    def flush_human() -> None:
        if not pending_human:
            return
        turns.append(
            ConversationTurn(
                id=f"journal-{max(pending_ids) if pending_ids else len(turns)}",
                role="user",
                content="\n\n".join(pending_human),
                journal_ids=tuple(pending_ids),
            )
        )
        pending_human.clear()
        pending_ids.clear()

    for journal in _journals(issue):
        notes = _journal_notes(journal)
        if not notes:
            continue
        journal_id = _journal_id(journal)
        if _journal_user_id(journal) in ai_user_ids:
            flush_human()
            turns.append(
                ConversationTurn(
                    id=f"journal-{journal_id}",
                    role="assistant",
                    content=notes,
                    journal_ids=(journal_id,),
                )
            )
        else:
            pending_human.append(notes)
            pending_ids.append(journal_id)
    flush_human()
    return turns


def _conversation_turns(state: TicketState) -> list[ConversationTurn]:
    raw_turns = state.get("recent_turns")
    if isinstance(raw_turns, list):
        return [
            ConversationTurn.from_dict(item)
            for item in raw_turns
            if isinstance(item, dict) and item.get("content")
        ]
    turns: list[ConversationTurn] = []
    for index, message in enumerate(state.get("messages", []), 1):
        turns.append(
            ConversationTurn(
                id=f"legacy-turn-{index}",
                role="assistant" if isinstance(message, AIMessage) else "user",
                content=str(message.content),
            )
        )
    return turns


def _working_memory(
    state: TicketState, *, issue: dict[str, Any] | None = None
) -> WorkingMemory:
    return WorkingMemory(
        issue=_working_issue(issue or dict(state.get("issue", {}))),
        current_plan=(
            dict(state["current_plan"])
            if isinstance(state.get("current_plan"), dict)
            else None
        ),
        plan_steps=tuple(
            dict(item) for item in state.get("plan_steps", []) if isinstance(item, dict)
        ),
        run_status=str(state.get("run_status") or "initialized"),
        waiting_reason=(
            str(state["waiting_reason"])
            if state.get("waiting_reason") is not None
            else None
        ),
        active_artifacts={
            key: dict(value)
            for key, value in state.get("active_artifacts", {}).items()
            if isinstance(value, dict)
        },
    )


def _artifact_refs(state: TicketState) -> list[ArtifactRef]:
    raw = state.get("artifact_refs")
    if not isinstance(raw, list):
        raw = state.get("artifacts", [])
    refs: list[ArtifactRef] = []
    for item in raw:
        if not isinstance(item, dict) or "artifact_id" not in item:
            continue
        try:
            refs.append(ArtifactRef.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return _deduplicate_refs(refs)


def _deduplicate_refs(refs: list[ArtifactRef]) -> list[ArtifactRef]:
    result: list[ArtifactRef] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.artifact_id in seen:
            continue
        seen.add(ref.artifact_id)
        result.append(ref)
    return result


def _latest_turn_id(state: TicketState) -> str | None:
    turns = _conversation_turns(state)
    return turns[-1].id if turns else None


def _selected_step_artifact_ids(
    state: TicketState, *, step: TaskStep, step_index: int
) -> tuple[str, ...]:
    selected = list(step.input_artifact_ids)
    plan_steps = state.get("plan_steps", [])
    for dependency in step.depends_on:
        dependency_index = dependency - 1
        if dependency_index < 0 or dependency_index >= step_index:
            raise ContextEngineError(
                f"step {step_index + 1} has an invalid dependency: {dependency}"
            )
        dependency_state = plan_steps[dependency_index]
        if dependency_state.get("status") != "completed":
            raise ContextEngineError(
                f"step {step_index + 1} depends on incomplete step {dependency}"
            )
        for artifact in dependency_state.get("artifacts", []):
            if isinstance(artifact, dict) and isinstance(artifact.get("artifact_id"), str):
                selected.append(artifact["artifact_id"])
    return tuple(dict.fromkeys(selected))


def _context_error_notes(exc: ContextEngineError) -> str:
    if isinstance(exc, ContextLimitExceeded):
        return (
            "入力がモデルのcontext上限を超えています。対象を分割するか、参照する成果物を絞ってください。\n"
            f"入力サイズ内訳: {json.dumps(exc.breakdown, ensure_ascii=False, sort_keys=True)}"
        )
    return f"会話contextを安全に準備できませんでした。\n理由: {exc}"


def _canonical_result_text(result: SkillExecutionResult) -> str | None:
    if result.assistant_turn_text and result.assistant_turn_text.strip():
        return result.assistant_turn_text.strip()
    for event in reversed(result.events):
        if event.kind in ("final_review", "final_return") and event.notes:
            return event.notes.strip()
    return None


def _context_error_plan(exc: ContextEngineError) -> TaskPlan:
    notes = _context_error_notes(exc)
    return TaskPlan(
        decision="needs_user",
        reason=notes,
        user_request=notes,
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


def _working_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in issue.items()
        if key not in {"journals", "description", "selected_artifacts"}
    }


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
