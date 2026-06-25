from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Protocol

from langchain_core.tools import BaseTool

from taskboard_agent.llm import LLMResponse
from taskboard_agent.skill_runtime import (
    GenericSkillRunner,
    ScriptedSkillRunner,
    SkillAgentPort,
    SkillEventSink,
    SkillEvent,
    SkillExecutionResult,
    _with_tool_artifacts,
    llm_response_events,
)
from taskboard_agent.skills import Skill, SkillRegistry
from taskboard_agent.structured_output import (
    generic_execution_response_format,
    task_plan_response_format,
    tool_execution_response_format,
)
from taskboard_agent.tools import (
    ToolExecutionError,
    ToolExecutionResult,
    execute_tool,
    require_tools_registered,
    tool_by_name,
)


PlanDecision = Literal["use_skill", "use_tools", "no_skill", "needs_user"]
StepKind = Literal["skill", "tool", "llm", "unavailable"]
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

    def tools_for(self, tool_names: tuple[str, ...] | list[str]) -> list[BaseTool]:
        ...


@dataclass(frozen=True)
class TaskStep:
    kind: StepKind
    purpose: str
    name: str | None = None
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskPlan:
    decision: PlanDecision
    reason: str
    skill_name: str | None = None
    tool_names: tuple[str, ...] = ()
    target_url: str | None = None
    task_input: dict[str, Any] | None = None
    user_request: str | None = None
    steps: tuple[TaskStep, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskStepExecution:
    index: int
    step: TaskStep
    result: SkillExecutionResult
    events: tuple[SkillEvent, ...]
    artifacts: tuple[dict[str, Any], ...] = ()
    context_messages: tuple[dict[str, Any], ...] = ()
    terminal_status: str | None = None

    @property
    def should_stop(self) -> bool:
        return self.terminal_status is not None


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
                    "ただし実行計画は必ず作業ステップに分解し、各ステップをskill/tool/llm/unavailableのいずれかに分類してください。"
                    "専用toolやskillがない作業でも、LLMの推論・要約・分析・比較・提案・文章作成で進められる場合はllmステップとして計画してください。"
                    "tool/skillステップのnameには、利用可能一覧にある機械名だけを正確に入れ、説明文や表示名を混ぜないでください。"
                    "実行できないことはunavailableステップまたはlimitationsに明示し、それ以外の実行可能な作業は止めずに進める計画にしてください。"
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
                plan = normalize_task_plan_names(plan, skills=skills, tools=tools)
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
        tools: list[BaseTool],
        tool_names: tuple[str, ...],
        task_input: dict[str, Any] | None = None,
        dry_run: bool = False,
        emit_event: SkillEventSink | None = None,
    ) -> SkillExecutionResult:
        require_tools_registered(tools, tool_names)
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
            return_after_tool_names=_return_after_tool_names(tool_names),
        )
        try:
            result = _parse_tool_execution(
                agent_result.final_text,
                tool_names=tool_names,
                dry_run=dry_run,
            )
        except TaskPlanningError:
            result = _fallback_tool_execution_from_artifacts(
                agent_result.tool_results,
                tool_names=tool_names,
                dry_run=dry_run,
            )
            if result is None:
                raise
        result = _with_tool_artifacts(result, agent_result.tool_results)
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

    def plan_notes(self, plan: TaskPlan) -> str:
        return _plan_notes(plan)

    def step_context_messages(
        self,
        *,
        issue: dict[str, Any],
        conversation_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        messages = list(conversation_messages or [])
        messages.append(
            {
                "role": "user",
                "content": (
                    "Redmineチケットの依頼内容:\n"
                    f"{json.dumps(_issue_context(issue), ensure_ascii=False)}"
                ),
            }
        )
        return messages

    def step_final_event(self, *, plan: TaskPlan, status: str) -> SkillEvent:
        final_notes = _step_final_notes(plan=plan, status=status)
        kind = "final_return" if status in ("failed",) else "final_review"
        return SkillEvent(kind, final_notes)

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
        skills, tools = self.planning_catalog()
        plan = normalize_task_plan_names(plan, skills=skills, tools=tools)
        execution_issue = dict(issue)
        if conversation_messages:
            execution_issue["conversation_context"] = conversation_messages
        if plan.steps:
            if emit_event is not None and announce_plan:
                emit_event(_plan_start_event(plan))
            result = self._execute_steps(
                issue=execution_issue,
                plan=plan,
                dry_run=dry_run,
                conversation_messages=conversation_messages,
            )
            if emit_event is not None:
                return _drop_start_event(result)
            if not announce_plan:
                return _drop_start_event(result)
            return _prepend_plan_event(result, _plan_notes(plan))
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
                tools = self._tool_catalog.tools_for(plan.tool_names)
            except (ToolExecutionError, RuntimeError) as exc:
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
                tools=tools,
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
            tools = self._tool_catalog.tools_for(skill.required_tools)
        except (ToolExecutionError, RuntimeError) as exc:
            return _needs_user_result(
                f"スキル `{plan.skill_name}` に必要なtoolが不足しています。\n理由: {exc}"
            )
        if skill.runner is not None:
            runner = ScriptedSkillRunner(skill=skill, tools=tools)
        else:
            runner = GenericSkillRunner(
                skill=skill,
                tools=tools,
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

    def _execute_steps(
        self,
        *,
        issue: dict[str, Any],
        plan: TaskPlan,
        dry_run: bool,
        conversation_messages: list[dict[str, Any]] | None,
    ) -> SkillExecutionResult:
        events: list[SkillEvent] = [SkillEvent("start", "作業を開始します。")]
        artifacts: list[dict[str, Any]] = []
        status = "dry_run" if dry_run else "processed"
        step_context = self.step_context_messages(
            issue=issue,
            conversation_messages=conversation_messages,
        )

        for index, step in enumerate(plan.steps, 1):
            execution = self.execute_single_step(
                issue=issue,
                plan=plan,
                step=step,
                step_index=index,
                dry_run=dry_run,
                step_context=step_context,
            )
            events.extend(execution.events)
            artifacts.extend(execution.artifacts)
            step_context.extend(execution.context_messages)
            if execution.should_stop:
                status = execution.terminal_status or status
                break

        events.append(self.step_final_event(plan=plan, status=status))
        return SkillExecutionResult(
            status=status,
            events=tuple(events),
            artifacts=tuple(artifacts),
            dry_run=dry_run,
        )

    def execute_single_step(
        self,
        *,
        issue: dict[str, Any],
        plan: TaskPlan,
        step: TaskStep,
        step_index: int,
        dry_run: bool = False,
        step_context: list[dict[str, Any]] | None = None,
    ) -> TaskStepExecution:
        if step_index < 1:
            raise ValueError("step_index must be 1-based")
        active_context = list(step_context or [])
        if step.kind == "unavailable":
            result = SkillExecutionResult(
                status="skipped",
                events=(),
                dry_run=dry_run,
            )
            return TaskStepExecution(
                index=step_index,
                step=step,
                result=result,
                events=(
                    SkillEvent(
                        "progress",
                        f"未実行の作業 {step_index}: {step.purpose}",
                    ),
                ),
            )

        try:
            if step.kind == "llm":
                result = self._run_llm_step(
                    issue=issue,
                    step=step,
                    plan=plan,
                    dry_run=dry_run,
                    step_context=active_context,
                )
            elif step.kind == "tool":
                result = self._run_tool_step(
                    issue=issue,
                    step=step,
                    plan=plan,
                    dry_run=dry_run,
                    step_context=active_context,
                )
            else:
                result = self._run_skill_step(issue=issue, step=step, dry_run=dry_run)
        except Exception as exc:
            result = _step_exception_result(step, exc, dry_run=dry_run)

        if result.status in ("failed", "missing_tool", "needs_user"):
            result = self._recover_step_failure(
                issue=issue,
                step=step,
                plan=plan,
                dry_run=dry_run,
                step_context=active_context,
                failed_result=result,
            )

        terminal_status = (
            result.status
            if result.status in ("failed", "missing_tool", "needs_user")
            else None
        )
        return TaskStepExecution(
            index=step_index,
            step=step,
            result=result,
            events=_step_events(step_index, step, result),
            artifacts=result.artifacts,
            context_messages=tuple(
                _result_context_messages(step_index, step, result)
            ),
            terminal_status=terminal_status,
        )

    def _run_tool_step(
        self,
        *,
        issue: dict[str, Any],
        step: TaskStep,
        plan: TaskPlan,
        dry_run: bool,
        step_context: list[dict[str, Any]],
    ) -> SkillExecutionResult:
        if not step.name:
            return _missing_step_name_result("tool", step)
        try:
            tools = self._tool_catalog.tools_for((step.name,))
        except (ToolExecutionError, RuntimeError) as exc:
            return _needs_user_result(f"必要なtool `{step.name}` が不足しています。\n理由: {exc}")
        selected_tool = tool_by_name(tools, step.name)
        arguments, repair_events = _repair_tool_step_arguments(
            tool=selected_tool,
            step=step,
            plan=plan,
            issue=issue,
            step_context=step_context,
        )
        arguments, schema_events = _normalize_tool_arguments_for_schema(
            tool=selected_tool,
            tool_name=step.name,
            arguments=arguments,
        )
        try:
            tool_result = execute_tool(
                selected_tool,
                arguments,
                allow_writes=not dry_run,
            )
        except (ToolExecutionError, RuntimeError) as exc:
            return SkillExecutionResult(
                status="failed",
                events=(SkillEvent("final_return", f"tool `{step.name}` の実行に失敗しました。\n理由: {exc}"),),
                dry_run=dry_run,
            )
        result = _fallback_tool_execution_from_artifacts(
            (tool_result,),
            tool_names=(step.name,),
            dry_run=dry_run,
        )
        if result is None:
            result = _tool_result_execution(tool_result, dry_run=dry_run)
        if repair_events or schema_events:
            result = _insert_after_start(result, (*repair_events, *schema_events))
        return _with_tool_artifacts(result, (tool_result,))

    def _run_skill_step(
        self,
        *,
        issue: dict[str, Any],
        step: TaskStep,
        dry_run: bool,
    ) -> SkillExecutionResult:
        if not step.name:
            return _missing_step_name_result("skill", step)
        skills = {skill.name: skill for skill in self._skill_registry.list()}
        skill = skills.get(step.name)
        if skill is None:
            return _needs_user_result(f"必要なスキル `{step.name}` が登録されていません。")
        try:
            tools = self._tool_catalog.tools_for(skill.required_tools)
        except (ToolExecutionError, RuntimeError) as exc:
            return _needs_user_result(f"スキル `{step.name}` に必要なtoolが不足しています。\n理由: {exc}")
        runner = (
            ScriptedSkillRunner(skill=skill, tools=tools)
            if skill.runner is not None
            else GenericSkillRunner(
                skill=skill,
                tools=tools,
                skill_agent=self._skill_agent,
            )
        )
        return runner.run(
            issue=issue,
            task_input=step.arguments or {},
            dry_run=dry_run,
        )

    def _run_llm_step(
        self,
        *,
        issue: dict[str, Any],
        step: TaskStep,
        plan: TaskPlan,
        dry_run: bool,
        step_context: list[dict[str, Any]],
    ) -> SkillExecutionResult:
        return self._generic_runner.run(
            issue=issue,
            dry_run=dry_run,
            conversation_messages=[
                *step_context,
                {
                    "role": "user",
                    "content": (
                        "次の作業ステップをLLMで実行してください。\n"
                        f"ステップ目的: {step.purpose}\n"
                        f"補足引数: {json.dumps(step.arguments or {}, ensure_ascii=False)}"
                    ),
                },
            ],
            task_plan=TaskPlan(
                decision="no_skill",
                reason=step.purpose,
                task_input={"instruction": step.purpose},
                limitations=plan.limitations,
            ),
        )

    def _recover_step_failure(
        self,
        *,
        issue: dict[str, Any],
        step: TaskStep,
        plan: TaskPlan,
        dry_run: bool,
        step_context: list[dict[str, Any]],
        failed_result: SkillExecutionResult,
    ) -> SkillExecutionResult:
        analysis_event = SkillEvent(
            "progress",
            (
                "ステップの失敗原因を確認します。\n"
                f"対象: {step.kind} {step.name or ''}\n"
                f"失敗内容:\n{_event_notes(failed_result.events)}"
            ),
        )
        if step.kind != "tool":
            return _append_events(
                failed_result,
                (
                    analysis_event,
                    SkillEvent(
                        "progress",
                        "このステップは自動で再実行できる修正方法を特定できませんでした。",
                    ),
                ),
            )

        retry_result = self._retry_tool_step_after_failure(
            issue=issue,
            step=step,
            plan=plan,
            dry_run=dry_run,
            step_context=step_context,
            failed_result=failed_result,
        )
        if retry_result is None:
            return _append_events(
                failed_result,
                (
                    analysis_event,
                    SkillEvent(
                        "progress",
                        "toolスキーマと文脈を確認しましたが、自動補正できる引数変更を特定できませんでした。",
                    ),
                ),
            )
        return _merge_recovery_result(
            failed_result,
            (
                analysis_event,
                SkillEvent("progress", "失敗原因を踏まえて引数を補正し、toolを再実行します。"),
            ),
            retry_result,
        )

    def _retry_tool_step_after_failure(
        self,
        *,
        issue: dict[str, Any],
        step: TaskStep,
        plan: TaskPlan,
        dry_run: bool,
        step_context: list[dict[str, Any]],
        failed_result: SkillExecutionResult,
    ) -> SkillExecutionResult | None:
        if not step.name:
            return None
        try:
            tools = self._tool_catalog.tools_for((step.name,))
        except (ToolExecutionError, RuntimeError):
            return None
        selected_tool = tool_by_name(tools, step.name)
        repaired_context = [
            *step_context,
            {
                "role": "assistant",
                "content": f"直前の失敗内容:\n{_event_notes(failed_result.events)}",
            },
        ]
        arguments, repair_events = _repair_tool_step_arguments(
            tool=selected_tool,
            step=step,
            plan=plan,
            issue=issue,
            step_context=repaired_context,
        )
        arguments, schema_events = _normalize_tool_arguments_for_schema(
            tool=selected_tool,
            tool_name=step.name,
            arguments=arguments,
        )
        if arguments == (step.arguments or {}):
            return None
        try:
            tool_result = execute_tool(
                selected_tool,
                arguments,
                allow_writes=not dry_run,
            )
        except (ToolExecutionError, RuntimeError) as exc:
            return SkillExecutionResult(
                status="failed",
                events=(
                    *repair_events,
                    *schema_events,
                    SkillEvent(
                        "final_return",
                        f"補正後もtool `{step.name}` の実行に失敗しました。\n理由: {exc}",
                    ),
                ),
                dry_run=dry_run,
            )
        result = _fallback_tool_execution_from_artifacts(
            (tool_result,),
            tool_names=(step.name,),
            dry_run=dry_run,
        )
        if result is None:
            result = _tool_result_execution(tool_result, dry_run=dry_run)
        if repair_events or schema_events:
            result = _insert_after_start(result, (*repair_events, *schema_events))
        return _with_tool_artifacts(result, (tool_result,))


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
    steps = _parse_task_steps(
        _normalize_json_null(data.get("steps", [])),
        purpose_fallback=_task_step_purpose_fallback(
            reason=reason,
            task_input=task_input,
        ),
    )
    limitations = _parse_string_tuple(data.get("limitations", []), "limitations")
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
        user_request=_non_empty_str_or_none(user_request),
        steps=steps,
        limitations=limitations,
    )


def normalize_task_plan_names(
    plan: TaskPlan,
    *,
    skills: list[Skill],
    tools: list[dict[str, Any]],
) -> TaskPlan:
    skill_names = {skill.name for skill in skills}
    tool_names = {
        name
        for tool in tools
        if isinstance((name := tool.get("name")), str) and name
    }
    skill_name = _canonical_catalog_name(plan.skill_name, skill_names)
    normalized_tool_names = tuple(
        name
        for tool_name in plan.tool_names
        if (name := _canonical_catalog_name(tool_name, tool_names)) is not None
    )
    normalized_steps = tuple(
        _normalize_step_name(step, skill_names=skill_names, tool_names=tool_names)
        for step in plan.steps
    )
    if (
        skill_name == plan.skill_name
        and normalized_tool_names == plan.tool_names
        and normalized_steps == plan.steps
    ):
        return plan
    return replace(
        plan,
        skill_name=skill_name,
        tool_names=normalized_tool_names,
        steps=normalized_steps,
    )


def _normalize_step_name(
    step: TaskStep,
    *,
    skill_names: set[str],
    tool_names: set[str],
) -> TaskStep:
    if step.kind == "skill":
        name = _canonical_catalog_name(step.name, skill_names)
    elif step.kind == "tool":
        name = _canonical_catalog_name(step.name, tool_names)
    else:
        name = step.name
    return step if name == step.name else replace(step, name=name)


def _canonical_catalog_name(value: str | None, catalog_names: set[str]) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if text in catalog_names:
        return text
    matches = [
        name
        for name in catalog_names
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
    ]
    if len(matches) == 1:
        return matches[0]
    return text


def _repair_tool_step_arguments(
    *,
    tool: BaseTool,
    step: TaskStep,
    plan: TaskPlan,
    issue: dict[str, Any],
    step_context: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[SkillEvent]]:
    arguments = dict(step.arguments or {})
    schema = _tool_input_schema(tool)
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list) or not isinstance(properties, dict):
        return arguments, []

    events: list[SkillEvent] = []
    for field in required:
        if not isinstance(field, str) or field in arguments:
            continue
        property_schema = properties.get(field)
        if field == "query" and _schema_accepts_string(property_schema):
            query = _infer_query_argument(
                step=step,
                plan=plan,
                issue=issue,
                step_context=step_context,
            )
            if query:
                arguments[field] = query
                events.append(
                    SkillEvent(
                        "progress",
                        f"tool `{step.name}` の不足引数 `{field}` を文脈から補完しました: {query}",
                    )
                )
    return arguments, events


def _normalize_tool_arguments_for_schema(
    *,
    tool: BaseTool,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], list[SkillEvent]]:
    schema = _tool_input_schema(tool)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return arguments, []

    normalized = dict(arguments)
    events: list[SkillEvent] = []
    unknown = [key for key in normalized if key not in properties]
    for key in unknown:
        normalized.pop(key, None)
    if unknown:
        events.append(
            SkillEvent(
                "progress",
                f"tool `{tool_name}` のスキーマにない引数を除去しました: {', '.join(unknown)}",
            )
        )

    for key, value in list(normalized.items()):
        property_schema = properties.get(key)
        if not isinstance(property_schema, dict):
            continue
        coerced = _coerce_schema_value(value, property_schema.get("type"))
        if coerced is not _NO_COERCION and coerced != value:
            normalized[key] = coerced
            events.append(
                SkillEvent(
                    "progress",
                    f"tool `{tool_name}` の引数 `{key}` の型を補正しました。",
                )
            )
    return normalized, events


_NO_COERCION = object()


def _coerce_schema_value(value: Any, expected: Any) -> Any:
    if expected is None:
        return _NO_COERCION
    expected_types = expected if isinstance(expected, list) else [expected]
    if "string" in expected_types and not isinstance(value, str):
        return str(value)
    if "integer" in expected_types and isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
    if "number" in expected_types and isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)", stripped):
            return float(stripped)
    if "boolean" in expected_types and isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in ("true", "yes", "1"):
            return True
        if stripped in ("false", "no", "0"):
            return False
    return _NO_COERCION


def _schema_accepts_string(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return True
    expected = schema.get("type")
    if expected is None:
        return True
    expected_types = expected if isinstance(expected, list) else [expected]
    return "string" in expected_types


def _tool_input_schema(tool: BaseTool) -> dict[str, Any]:
    args_schema = tool.args_schema
    if isinstance(args_schema, dict):
        return args_schema
    if hasattr(args_schema, "model_json_schema"):
        schema = args_schema.model_json_schema()
        return schema if isinstance(schema, dict) else {}
    return {"type": "object", "properties": dict(tool.args)}


def _infer_query_argument(
    *,
    step: TaskStep,
    plan: TaskPlan,
    issue: dict[str, Any],
    step_context: list[dict[str, Any]],
) -> str | None:
    for value in _argument_alias_values(step.arguments):
        if query := _clean_query_candidate(value):
            return query
    for source in _query_context_sources(
        step=step,
        plan=plan,
        issue=issue,
        step_context=step_context,
    ):
        if query := _extract_search_query(source):
            return query
    return None


def _argument_alias_values(arguments: dict[str, Any] | None) -> list[Any]:
    if not isinstance(arguments, dict):
        return []
    aliases = ("query", "search_query", "keyword", "keywords", "q", "term")
    return [arguments[key] for key in aliases if key in arguments]


def _query_context_sources(
    *,
    step: TaskStep,
    plan: TaskPlan,
    issue: dict[str, Any],
    step_context: list[dict[str, Any]],
) -> list[str]:
    sources: list[str] = []
    for message in reversed(step_context):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            sources.append(content)
    sources.extend(
        item
        for item in (
            step.purpose,
            plan.reason,
            plan.user_request,
            _task_input_text(plan.task_input),
            str(issue.get("subject") or ""),
            str(issue.get("description") or ""),
            json.dumps(issue.get("conversation_context"), ensure_ascii=False),
        )
        if isinstance(item, str) and item.strip()
    )
    return sources


def _task_input_text(task_input: dict[str, Any] | None) -> str:
    if not isinstance(task_input, dict):
        return ""
    parts: list[str] = []
    for key in ("query", "search_query", "keyword", "keywords", "instruction", "target_url"):
        value = task_input.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts)


def _extract_search_query(text: str) -> str | None:
    for pattern in (
        r"(?:検索キーワード|検索語|検索ワード|キーワード|query)\s*(?:[は:：=]|として|に)?\s*[「『\"`]([^」』\"`\n。]+)[」』\"`]",
        r"[「『\"`]([^」』\"`\n]{2,80})[」』\"`]\s*(?:を用いて|で|について)?\s*(?:Web検索|検索|調査)",
        r"(?:検索キーワード|検索語|検索ワード|キーワード|query)\s*(?:[は:：=]|として|に)\s*([^」』\"`\n。]+)",
        r"(?:Web検索|検索|調査)\s*(?:する|を実行する|します)?\s*(?:キーワード|対象)?\s*[「『\"`]?([^」』\"`\n。]+)",
        r"([^。\n]{2,80})\s*を(?:Web検索|検索|調査)(?:する|します|してください|して)?",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if query := _clean_query_candidate(match.group(1)):
                return query
    return None


def _clean_query_candidate(value: Any) -> str | None:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if str(item).strip())
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    candidate = candidate.strip(" \t\r\n:：=「」『』\"'`")
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = re.sub(r"(?:を)?(?:Web検索|検索|調査)(?:する|します|してください)?$", "", candidate).strip()
    candidate = candidate.strip(" \t\r\n:：=「」『』\"'`")
    if not candidate or len(candidate) > 120:
        return None
    if candidate in {"web_search_pages", "Web検索", "検索", "調査"}:
        return None
    if re.fullmatch(r".*(?:準備|選定|決定|実行|取得|収集)(?:する|します)?", candidate):
        return None
    if "web_search_pages" in candidate:
        return None
    return candidate


def _parse_task_steps(
    value: Any,
    *,
    purpose_fallback: str | None = None,
) -> tuple[TaskStep, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TaskPlanningError("task plan steps must be an array")
    steps: list[TaskStep] = []
    for item in value:
        if not isinstance(item, dict):
            raise TaskPlanningError("task plan step must be an object")
        kind = item.get("kind")
        if kind not in ("skill", "tool", "llm", "unavailable"):
            raise TaskPlanningError("task plan step missing valid kind")
        purpose = item.get("purpose")
        if not isinstance(purpose, str):
            raise TaskPlanningError("task plan step missing purpose")
        normalized_purpose = purpose.strip() or purpose_fallback
        if not normalized_purpose:
            raise TaskPlanningError("task plan step missing purpose")
        name = _normalize_json_null(item.get("name"))
        if name is not None and not isinstance(name, str):
            raise TaskPlanningError("task plan step name must be a string or null")
        arguments = _normalize_json_null(item.get("arguments"))
        if arguments is not None and not isinstance(arguments, dict):
            raise TaskPlanningError("task plan step arguments must be an object or null")
        steps.append(
            TaskStep(
                kind=kind,
                purpose=normalized_purpose,
                name=name.strip() if isinstance(name, str) else None,
                arguments=arguments,
            )
        )
    return tuple(steps)


def _task_step_purpose_fallback(
    *,
    reason: Any,
    task_input: Any,
) -> str | None:
    if isinstance(task_input, dict):
        instruction = _non_empty_str_or_none(task_input.get("instruction"))
        if instruction:
            return instruction
    return _non_empty_str_or_none(reason)


def _non_empty_str_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _parse_string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TaskPlanningError(f"task plan {label} must be a string array")
    return tuple(item.strip() for item in value if item.strip())


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
    if plan.steps:
        for step in plan.steps:
            if step.kind == "skill" and step.name not in skill_names:
                raise TaskPlanningError(f"step referenced unknown skill: {step.name}")
            if step.kind == "tool" and step.name not in tool_names:
                raise TaskPlanningError(f"step referenced unknown tool: {step.name}")
            if step.kind in ("skill", "tool") and not step.name:
                raise TaskPlanningError(f"{step.kind} step requires name")
        return
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
    repair_hint = _plan_retry_repair_hint(str(error))
    return (
        "前回の出力は次の検証エラーにより無効です。\n"
        f"- {error}\n\n"
        f"{repair_hint}\n"
        "作業内容の判断は変えず、型と分岐の整合性だけを修正してください。\n"
        "空文字ではなく、実行内容が分かる短い値を入れてください。\n"
        "APIで指定された出力構造と分岐条件に従って再回答してください。"
    )


def _plan_retry_repair_hint(error: str) -> str:
    if "step missing purpose" in error:
        return (
            "修正指示: 各steps[].purposeには、そのstepで実行する作業内容を説明する非空文字列を入れてください。"
        )
    if "missing reason" in error:
        return "修正指示: reasonには判断理由を説明する非空文字列を入れてください。"
    if "needs_user requires a non-empty user_request" in error:
        return "修正指示: user_requestにはユーザーへ確認したい内容を非空文字列で入れてください。"
    if "requires name" in error:
        return "修正指示: skill/toolステップのnameには、利用可能一覧にある機械名を正確に入れてください。"
    return "修正指示: エラーに示されたフィールドを、空文字や矛盾がない有効な値へ修正してください。"


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


def _tool_result_execution(
    tool_result: ToolExecutionResult,
    *,
    dry_run: bool,
) -> SkillExecutionResult:
    content = tool_result.content
    ok = content.get("ok")
    if ok is False:
        status = "failed"
        event = SkillEvent(
            "final_return",
            f"tool `{tool_result.name}` の実行に失敗しました。\n理由: {content.get('error') or '不明なエラー'}",
        )
    else:
        status = "dry_run" if dry_run else "processed"
        event = SkillEvent(
            "final_review",
            f"tool `{tool_result.name}` を実行しました。\n結果:\n{json.dumps(content, ensure_ascii=False)}",
        )
    return SkillExecutionResult(
        status=status,
        events=(
            SkillEvent("start", "作業を開始します。"),
            SkillEvent("progress", f"toolを実行しました: {tool_result.name}"),
            event,
        ),
        dry_run=dry_run,
    )


def _fallback_tool_execution_from_artifacts(
    tool_results: tuple[Any, ...],
    *,
    tool_names: tuple[str, ...],
    dry_run: bool,
) -> SkillExecutionResult | None:
    for result in tool_results:
        content = getattr(result, "content", None)
        if not isinstance(content, dict):
            continue
        if getattr(result, "name", None) == "web_search_pages" and content.get("ok") is False:
            error = content.get("error") or "不明なエラー"
            return SkillExecutionResult(
                status="failed",
                events=(
                    SkillEvent("start", "作業を開始します。"),
                    SkillEvent("progress", f"toolを実行しました: {', '.join(tool_names)}"),
                    SkillEvent("final_return", f"Web検索に失敗しました。\n理由: {error}"),
                ),
                dry_run=dry_run,
            )
        artifact = content.get("context_artifact")
        if isinstance(artifact, dict) and artifact.get("type") == "web_search_pages":
            status = "dry_run" if dry_run else "processed"
            return SkillExecutionResult(
                status=status,
                events=(
                    SkillEvent("start", "作業を開始します。"),
                    SkillEvent("progress", f"toolを実行しました: {', '.join(tool_names)}"),
                    SkillEvent(
                        "final_review",
                        (
                            "tool実行後の最終JSON生成が空または不正だったため、"
                            "取得済みの検索結果から報告を作成しました。"
                        ),
                    ),
                ),
                dry_run=dry_run,
            )
    return None


def _missing_step_name_result(kind: str, step: TaskStep) -> SkillExecutionResult:
    return SkillExecutionResult(
        status="missing_tool",
        events=(
            SkillEvent(
                "final_review",
                f"{kind}ステップの実行対象名が不足しています。\nステップ: {step.purpose}",
            ),
        ),
    )


def _step_exception_result(
    step: TaskStep,
    exc: Exception,
    *,
    dry_run: bool,
) -> SkillExecutionResult:
    return SkillExecutionResult(
        status="failed",
        events=(
            SkillEvent(
                "final_return",
                f"{step.kind}ステップの実行中に例外が発生しました。\n理由: {exc}",
            ),
        ),
        dry_run=dry_run,
    )


def _append_events(
    result: SkillExecutionResult,
    events: tuple[SkillEvent, ...],
) -> SkillExecutionResult:
    return SkillExecutionResult(
        status=result.status,
        events=(*result.events, *events),
        target_url=result.target_url,
        page_title=result.page_title,
        briefing=result.briefing,
        bookmark_url=result.bookmark_url,
        bookmark_payload=result.bookmark_payload,
        artifacts=result.artifacts,
        dry_run=result.dry_run,
    )


def _merge_recovery_result(
    failed_result: SkillExecutionResult,
    recovery_events: tuple[SkillEvent, ...],
    retry_result: SkillExecutionResult,
) -> SkillExecutionResult:
    retry_events = tuple(
        event for event in retry_result.events if event.kind != "start"
    )
    return SkillExecutionResult(
        status=retry_result.status,
        events=(*failed_result.events, *recovery_events, *retry_events),
        target_url=retry_result.target_url,
        page_title=retry_result.page_title,
        briefing=retry_result.briefing,
        bookmark_url=retry_result.bookmark_url,
        bookmark_payload=retry_result.bookmark_payload,
        artifacts=retry_result.artifacts,
        dry_run=retry_result.dry_run,
    )


def _step_events(
    index: int,
    step: TaskStep,
    result: SkillExecutionResult,
) -> tuple[SkillEvent, ...]:
    events = [SkillEvent("progress", f"ステップ {index} を実行しました: {step.purpose}")]
    for event in result.events:
        if event.kind == "start":
            continue
        events.append(event)
    return tuple(events)


def _result_context_messages(
    index: int,
    step: TaskStep,
    result: SkillExecutionResult,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": (
                f"ステップ {index} 実行結果:\n"
                f"目的: {step.purpose}\n"
                f"状態: {result.status}\n"
                f"報告:\n{_event_notes(result.events)}"
            ),
        }
    ]
    for artifact in result.artifacts:
        messages.append(
            {
                "role": "assistant",
                "content": f"ステップ {index} 成果JSON:\n{json.dumps(artifact, ensure_ascii=False)}",
            }
        )
    return messages


def _event_notes(events: tuple[SkillEvent, ...]) -> str:
    notes = [event.notes for event in events if event.notes]
    return "\n\n".join(notes) if notes else "(なし)"


def _step_final_notes(*, plan: TaskPlan, status: str) -> str:
    lines = ["計画した作業ステップの実行を終了しました。"]
    if plan.limitations:
        lines.append("\n実行できないこと・未確認事項:")
        lines.extend(f"- {item}" for item in plan.limitations)
    if status in ("failed", "missing_tool", "needs_user"):
        lines.append(f"\n終了状態: {status}")
    return "\n".join(lines)


def _return_after_tool_names(tool_names: tuple[str, ...]) -> set[str] | None:
    if "web_search_pages" in tool_names:
        return {"web_search_pages"}
    return None


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
        artifacts=result.artifacts,
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
        artifacts=result.artifacts,
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
        artifacts=result.artifacts,
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
    notes = (
        f"ユーザーから依頼された作業は、{target}だと理解しました。"
        f"{plan.reason}と判断したためです。{method}"
    )
    if plan.steps:
        step_lines = [
            f"{index}. {step.kind}: {step.purpose}"
            for index, step in enumerate(plan.steps, 1)
        ]
        notes = f"{notes}\n\n作業ステップ:\n" + "\n".join(step_lines)
    if plan.limitations:
        notes = (
            f"{notes}\n\n実行できないこと・未確認事項:\n"
            + "\n".join(f"- {item}" for item in plan.limitations)
        )
    return notes


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
        "ただし、ユーザー依頼は必ずstepsへ作業ステップとして分解してください。"
        "各ステップは kind=skill/tool/llm/unavailable のいずれかにします。"
        "検索・取得・抽出などはtool、登録済み手順に合うものはskill、要約・分類・比較・分析・提案・判断・文章作成はllmにしてください。"
        "toolやskillがないことを理由に、LLMで進められる工程を未対応扱いしないでください。"
        "どうしても実行できない工程だけunavailableにし、limitationsにも理由を書いてください。\n"
        "依頼されていない作業を含むスキルは選ばないでください。例えば「Webページ本文を取得して」だけなら、ブックマーク登録まで行うスキルは使わずfetch_web_pageだけを使ってください。\n"
        "分岐規則:\n"
        "- use_skillではskill_nameを指定し、tool_namesは必ず[]にする。\n"
        "- use_toolsではskill_nameをnullにし、tool_namesを1件以上指定する。\n"
        "- no_skillではskill_nameをnull、tool_namesを[]にする。\n"
        "- needs_userではskill_nameをnull、tool_namesを[]にし、user_requestを指定する。\n"
        "steps/limitations規則:\n"
        "- stepsには作業順にステップを入れる。実行可能な部分を省略しない。\n"
        "- toolステップはnameに利用可能なtoolのnameを正確に入れる。説明文、表示名、括弧付き表記を混ぜない。\n"
        "- skillステップはnameに利用可能なskillのnameを正確に入れる。説明文、表示名、括弧付き表記を混ぜない。\n"
        "- ユーザーが `web_search_pages` のようなtool名を明示した場合、nameにはその文字列だけを入れる。\n"
        "- argumentsには実行引数を入れる。\n"
        "- llmステップはname=null、purposeにLLMで行う作業を具体的に入れる。\n"
        "- unavailableステップはname=null、purposeに実行できない作業と理由を入れる。\n"
        "- limitationsには未確認事項、外部制約、実行できないことを短く列挙する。\n"
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
                "`web_search_pages`を使った場合は、検索結果一覧と各URLの本文取得ステータス"
                "（正常またはエラー、エラー理由）を最終報告に必ず含めてください。"
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
