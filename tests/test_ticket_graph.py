from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from taskboard_agent.llm import LLMResponse
from taskboard_agent.skill_runtime import SkillEvent, SkillExecutionResult
from taskboard_agent.task_executor import TaskPlan, TaskStep, TaskStepExecution
from taskboard_agent.ticket_graph import TicketConversationGraph


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], **_kwargs: Any) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self.responses.pop(0))


class FakeOrchestrator:
    def __init__(
        self,
        *,
        plan: TaskPlan | None = None,
        fail_revision_once: bool = False,
        artifacts: tuple[dict[str, Any], ...] = (),
        tools: list[dict[str, Any]] | None = None,
        step_statuses: list[str] | None = None,
    ) -> None:
        self.plan = plan or TaskPlan(
            decision="no_skill",
            reason="チケット本文だけで対応可能",
        )
        self.executions: list[dict[str, Any]] = []
        self.fail_revision_once = fail_revision_once
        self.artifacts = artifacts
        self.tools = tools or []
        self.step_statuses = list(step_statuses or [])

    def create_plan(self, issue: dict[str, Any]) -> TaskPlan:
        return self.plan

    def planning_catalog(self) -> tuple[list[Any], list[dict[str, Any]]]:
        return [], self.tools

    def execute_plan(
        self,
        *,
        issue: dict[str, Any],
        plan: TaskPlan,
        dry_run: bool = False,
        emit_event: Any = None,
        announce_plan: bool = True,
        conversation_messages: list[dict[str, Any]] | None = None,
    ) -> SkillExecutionResult:
        self.executions.append(
            {
                "issue": issue,
                "plan": plan,
                "dry_run": dry_run,
                "announce_plan": announce_plan,
                "conversation_messages": conversation_messages,
            }
        )
        if announce_plan and emit_event is not None:
            emit_event(SkillEvent("start", "初回作業を計画しました。"))
        if not announce_plan and self.fail_revision_once:
            self.fail_revision_once = False
            raise RuntimeError("revision execution failed")
        return SkillExecutionResult(
            status="processed",
            events=(SkillEvent("final_review", "作業結果を確認してください。"),),
            target_url="https://example.test/article",
            page_title="Article",
            briefing="保存済みの要約本文",
            bookmark_url="https://bookmark.test/links/1",
            artifacts=self.artifacts,
        )

    def plan_notes(self, plan: TaskPlan) -> str:
        if plan.steps:
            step_lines = [
                f"{index}. {step.kind}: {step.purpose}"
                for index, step in enumerate(plan.steps, 1)
            ]
            return "初回作業を計画しました。\n\n作業ステップ:\n" + "\n".join(step_lines)
        return "初回作業を計画しました。"

    def step_context_messages(
        self,
        *,
        issue: dict[str, Any],
        conversation_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            *(conversation_messages or []),
            {"role": "user", "content": f"issue={issue['id']}"},
        ]

    def step_final_event(self, *, plan: TaskPlan, status: str) -> SkillEvent:
        kind = "final_return" if status == "failed" else "final_review"
        return SkillEvent(kind, "計画した作業ステップの実行を終了しました。")

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
        is_revision = any(
            "差し戻し" in str(message.get("content", ""))
            for message in (step_context or [])
        )
        self.executions.append(
            {
                "issue": issue,
                "plan": plan,
                "step": step,
                "step_index": step_index,
                "dry_run": dry_run,
                "step_context": step_context,
                "conversation_messages": step_context,
                "announce_plan": not is_revision,
            }
        )
        if self.fail_revision_once and is_revision:
            self.fail_revision_once = False
            raise RuntimeError("revision execution failed")
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
        status = self.step_statuses.pop(0) if self.step_statuses else "processed"
        if status in ("needs_user", "failed", "missing_tool"):
            event_kind = "final_return" if status == "failed" else "final_review"
            result = SkillExecutionResult(
                status=status,
                events=(SkillEvent(event_kind, "追加情報が必要です。"),),
                dry_run=dry_run,
            )
            return TaskStepExecution(
                index=step_index,
                step=step,
                result=result,
                events=(
                    SkillEvent(
                        "progress",
                        f"ステップ {step_index} を実行しました: {step.purpose}",
                    ),
                    *result.events,
                ),
                terminal_status=status,
            )
        result = SkillExecutionResult(
            status="processed",
            events=(SkillEvent("final_review", "作業結果を確認してください。"),),
            target_url="https://example.test/article",
            page_title="Article",
            briefing="保存済みの要約本文",
            bookmark_url="https://bookmark.test/links/1",
            artifacts=self.artifacts,
            dry_run=dry_run,
        )
        return TaskStepExecution(
            index=step_index,
            step=step,
            result=result,
            events=(SkillEvent("progress", f"ステップ {step_index} を実行しました: {step.purpose}"), *result.events),
            artifacts=self.artifacts,
            context_messages=(
                {
                    "role": "assistant",
                    "content": (
                        f"ステップ {step_index} 実行結果:\n"
                        "保存済みの要約本文"
                    ),
                },
                *(
                    {
                        "role": "assistant",
                        "content": f"ステップ {step_index} 成果JSON:\n{artifact}",
                    }
                    for artifact in self.artifacts
                ),
            ),
        )


def _issue(*, journals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": 123,
        "subject": "文章を作成する",
        "description": "案内文を作成してください。",
        "author": {"id": 7},
        "journals": journals or [],
    }


def _revision_response() -> str:
    return (
        '{"previous_work_summary":"案内文を作成した",'
        '"feedback_summary":"文章が長い",'
        '"requested_changes":[["短くする"]],'
        '"keep_existing_results":["案内文の主旨"],'
        '"work_to_redo":["案内文を短く修正する"],'
        '"task_plan":{"decision":"no_skill","reason":"文章の修正だけで完了する",'
        '"skill_name":null,"tool_names":[],"target_url":null,'
        '"task_input":"案内文を短くする","user_request":null}}'
    )


def test_initial_run_creates_ticket_conversation_and_interrupts() -> None:
    graph = TicketConversationGraph(
        task_orchestrator=FakeOrchestrator(),  # type: ignore[arg-type]
        llm=FakeLLM([]),
        checkpointer=InMemorySaver(),
        ai_user_id=42,
    )
    events: list[SkillEvent] = []

    result = graph.run(issue=_issue(), emit_event=events.append)

    assert result.status == "processed"
    assert events[0].kind == "start"
    assert "初回作業を計画しました。" in (events[0].notes or "")
    assert SkillEvent("progress", "作業結果を確認してください。") in events
    assert events[-1] == SkillEvent("final_review", "作業が終了しました。")
    state = graph.conversation_state(123)
    assert isinstance(state["messages"][0], HumanMessage)
    assert "案内文を作成してください" in state["messages"][0].content
    assert isinstance(state["messages"][-1], AIMessage)


def test_initial_plan_steps_are_saved_in_graph_state() -> None:
    graph = TicketConversationGraph(
        task_orchestrator=FakeOrchestrator(  # type: ignore[arg-type]
            plan=TaskPlan(
                decision="no_skill",
                reason="本文を整理して回答できる",
                steps=(
                    TaskStep(
                        kind="llm",
                        purpose="依頼内容を整理する",
                        arguments={"format": "bullets"},
                    ),
                    TaskStep(
                        kind="unavailable",
                        purpose="外部承認が必要な作業は実行しない",
                    ),
                ),
                limitations=("外部承認は未取得",),
            )
        ),
        llm=FakeLLM([]),
        checkpointer=InMemorySaver(),
        ai_user_id=42,
    )

    graph.run(issue=_issue())

    state = graph.conversation_state(123)
    assert state["run_status"] == "processed"
    assert state["current_step_index"] is None
    assert [result["status"] for result in state["step_results"]] == [
        "processed",
        "skipped",
    ]
    assert state["step_context"]["decision"] == "no_skill"
    assert state["step_context"]["reason"] == "本文を整理して回答できる"
    assert state["step_context"]["limitations"] == ["外部承認は未取得"]
    assert state["step_context"]["execution_model"] == "langgraph_step_loop"
    assert state["step_context"]["last_result_status"] == "processed"
    assert state["plan_steps"] == [
        {
            "id": "step-1",
            "index": 0,
            "kind": "llm",
            "name": None,
            "purpose": "依頼内容を整理する",
            "arguments": {"format": "bullets"},
            "status": "completed",
            "result": {
                "status": "processed",
                "target_url": "https://example.test/article",
                "page_title": "Article",
                "briefing": "保存済みの要約本文",
                "bookmark_url": "https://bookmark.test/links/1",
                "bookmark_payload": None,
                "artifacts": [],
                "dry_run": False,
            },
            "error": None,
            "artifacts": [
                {
                    "target_url": "https://example.test/article",
                    "page_title": "Article",
                    "briefing": "保存済みの要約本文",
                    "bookmark_url": "https://bookmark.test/links/1",
                }
            ],
        },
        {
            "id": "step-2",
            "index": 1,
            "kind": "unavailable",
            "name": None,
            "purpose": "外部承認が必要な作業は実行しない",
            "arguments": None,
            "status": "skipped",
            "result": {
                "status": "skipped",
                "target_url": None,
                "page_title": None,
                "briefing": None,
                "bookmark_url": None,
                "bookmark_payload": None,
                "artifacts": [],
                "dry_run": False,
            },
            "error": None,
            "artifacts": [],
        },
    ]


def test_step_needs_user_keeps_stopped_step_and_pending_remainder() -> None:
    graph = TicketConversationGraph(
        task_orchestrator=FakeOrchestrator(  # type: ignore[arg-type]
            plan=TaskPlan(
                decision="no_skill",
                reason="本文を確認して回答する",
                steps=(
                    TaskStep(kind="llm", purpose="追加情報を確認する"),
                    TaskStep(kind="llm", purpose="回答を作成する"),
                ),
            ),
            step_statuses=["needs_user"],
        ),
        llm=FakeLLM([]),
        checkpointer=InMemorySaver(),
        ai_user_id=42,
    )
    events: list[SkillEvent] = []

    result = graph.run(issue=_issue(), emit_event=events.append)

    assert result.status == "needs_user"
    assert events[-1] == SkillEvent("final_review", "計画した作業ステップの実行を終了しました。")
    state = graph.conversation_state(123)
    assert state["current_step_index"] == 0
    assert state["run_status"] == "needs_user"
    assert [step["status"] for step in state["plan_steps"]] == [
        "needs_user",
        "pending",
    ]
    assert state["step_results"][0]["status"] == "needs_user"
    assert state["last_result"]["status"] == "needs_user"


def test_resume_adds_only_new_human_comment_and_publishes_revision_first() -> None:
    llm = FakeLLM([_revision_response()])
    orchestrator = FakeOrchestrator()
    graph = TicketConversationGraph(
        task_orchestrator=orchestrator,  # type: ignore[arg-type]
        llm=llm,
        checkpointer=InMemorySaver(),
        ai_user_id=42,
    )
    graph.run(issue=_issue())
    events: list[SkillEvent] = []
    resumed_issue = _issue(
        journals=[
            {"id": 1, "user": {"id": 42}, "notes": "作業結果を確認してください。"},
            {"id": 2, "user": {"id": 7}, "notes": "文章が長いので短くしてください。"},
        ]
    )

    result = graph.run(issue=resumed_issue, emit_event=events.append)

    assert result.status == "processed"
    assert events[0].kind == "start"
    assert "差し戻し内容を確認し、作業を再計画しました" in (events[0].notes or "")
    assert SkillEvent("progress", "作業結果を確認してください。") in events
    assert events[-1] == SkillEvent("final_review", "作業が終了しました。")
    assert orchestrator.executions[-1]["announce_plan"] is False
    assert orchestrator.executions[-1]["plan"].task_input == {
        "instruction": "案内文を短くする"
    }
    assert len(llm.calls) == 1
    state = graph.conversation_state(123)
    assert state["last_ingested_journal_id"] == 2
    assert state["feedback_analysis"]["requested_changes"] == ["短くする"]
    assert sum(
        "文章が長いので短くしてください" in str(message.content)
        for message in state["messages"]
    ) == 1

    duplicate_events: list[SkillEvent] = []
    duplicate_result = graph.run(
        issue=resumed_issue,
        emit_event=duplicate_events.append,
    )
    assert duplicate_result.status == "needs_user"
    assert len(llm.calls) == 1
    assert "修正内容をコメントしてください" in (duplicate_events[0].notes or "")


def test_sqlite_checkpoint_resumes_with_a_new_graph_instance(tmp_path: Any) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    first_orchestrator = FakeOrchestrator()
    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        first_graph = TicketConversationGraph(
            task_orchestrator=first_orchestrator,  # type: ignore[arg-type]
            llm=FakeLLM([]),
            checkpointer=checkpointer,
            ai_user_id=42,
        )
        first_graph.run(issue=_issue())

    second_orchestrator = FakeOrchestrator()
    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        second_graph = TicketConversationGraph(
            task_orchestrator=second_orchestrator,  # type: ignore[arg-type]
            llm=FakeLLM([_revision_response()]),
            checkpointer=checkpointer,
            ai_user_id=42,
        )
        second_graph.run(
            issue=_issue(
                journals=[
                    {"id": 10, "user": {"id": 7}, "notes": "短くしてください。"}
                ]
            )
        )

    assert len(second_orchestrator.executions) == 1
    assert second_orchestrator.executions[0]["announce_plan"] is False


def test_failed_revision_restarts_in_progress_before_execution() -> None:
    orchestrator = FakeOrchestrator(fail_revision_once=True)
    graph = TicketConversationGraph(
        task_orchestrator=orchestrator,  # type: ignore[arg-type]
        llm=FakeLLM([_revision_response()]),
        checkpointer=InMemorySaver(),
        ai_user_id=42,
    )
    graph.run(issue=_issue())
    resumed_issue = _issue(
        journals=[
            {"id": 1, "user": {"id": 7}, "notes": "短く修正してください。"}
        ]
    )

    with pytest.raises(RuntimeError, match="revision execution failed"):
        graph.run(issue=resumed_issue)

    events: list[SkillEvent] = []
    result = graph.run(issue=resumed_issue, emit_event=events.append)

    assert result.status == "processed"
    assert events[0].kind == "start"
    assert "中断した差し戻し作業を再開します" in (events[0].notes or "")
    assert events[-2].kind == "progress"
    assert events[-1] == SkillEvent("final_review", "作業が終了しました。")


def test_resume_question_is_planned_and_executed_from_conversation() -> None:
    response = (
        '{"previous_work_summary":"記事の要約と登録を完了した",'
        '"feedback_summary":"実施内容の説明を求められた",'
        '"requested_changes":[],"keep_existing_results":["登録済みブックマーク"],'
        '"work_to_redo":["作業履歴を確認する","実施内容を説明する"],'
        '"task_plan":{"decision":"no_skill","reason":"会話履歴だけで回答できる",'
        '"skill_name":null,"tool_names":[],"target_url":null,'
        '"task_input":null,"user_request":null}}'
    )
    llm = FakeLLM([response])
    orchestrator = FakeOrchestrator()
    graph = TicketConversationGraph(
        task_orchestrator=orchestrator,  # type: ignore[arg-type]
        llm=llm,
        checkpointer=InMemorySaver(),
        ai_user_id=42,
    )
    graph.run(issue=_issue())
    execution_count = len(orchestrator.executions)
    events: list[SkillEvent] = []

    result = graph.run(
        issue=_issue(
            journals=[
                {"id": 1, "user": {"id": 42}, "notes": "要約を作成しました。"},
                {"id": 2, "user": {"id": 7}, "notes": "作業を説明してください。"},
            ]
        ),
        emit_event=events.append,
    )

    assert result.status == "processed"
    assert len(orchestrator.executions) == execution_count + 1
    assert orchestrator.executions[-1]["plan"].decision == "no_skill"
    assert events[0].kind == "start"
    assert "作業を再計画しました" in (events[0].notes or "")
    assert SkillEvent("progress", "作業結果を確認してください。") in events
    assert events[-1] == SkillEvent("final_review", "作業が終了しました。")
    planner_messages = llm.calls[0]
    assert "保存済みの作業状態と成果物" not in str(planner_messages)
    assert any("保存済みの要約本文" in message["content"] for message in planner_messages)
    assert "差し戻されました" in planner_messages[-1]["content"]
    assert "作業を説明してください" in planner_messages[-1]["content"]
    execution_messages = orchestrator.executions[-1]["conversation_messages"]
    assert any("保存済みの要約本文" in message["content"] for message in execution_messages)
    state = graph.conversation_state(123)
    assert any(
        isinstance(message, AIMessage) and message.content == "要約を作成しました。"
        for message in state["messages"]
    )


def test_revision_plan_preserves_step_based_work() -> None:
    response = (
        '{"previous_work_summary":"初回作業を完了した",'
        '"feedback_summary":"追加調査と提案を求められた",'
        '"requested_changes":["OpenClawを調査して提案する"],'
        '"keep_existing_results":["既存コメント"],'
        '"work_to_redo":["Web検索する","検索結果をもとに提案する"],'
        '"task_plan":{"decision":"use_tools","reason":"検索後にLLMで提案できる",'
        '"skill_name":null,"tool_names":["Webページの情報収集 (web_search_pages)"],"target_url":null,'
        '"task_input":null,"user_request":null,'
        '"steps":['
        '{"kind":"tool","name":"Webページの情報収集 (web_search_pages)","purpose":"OpenClawを検索する",'
        '"arguments":{"query":"openclaw"}},'
        '{"kind":"llm","name":null,"purpose":"検索結果から企業内活用案を提案する",'
        '"arguments":null}],'
        '"limitations":["社内規程への適合は未確認"]}}'
    )
    llm = FakeLLM([response])
    orchestrator = FakeOrchestrator(
        tools=[{"name": "web_search_pages", "description": "Webページの情報収集"}]
    )
    graph = TicketConversationGraph(
        task_orchestrator=orchestrator,  # type: ignore[arg-type]
        llm=llm,
        checkpointer=InMemorySaver(),
        ai_user_id=42,
    )
    graph.run(issue=_issue())

    events: list[SkillEvent] = []
    graph.run(
        issue=_issue(
            journals=[
                {"id": 1, "user": {"id": 7}, "notes": "OpenClawを調査して提案してください。"}
            ]
        ),
        emit_event=events.append,
    )

    plan = orchestrator.executions[-1]["plan"]
    assert plan.steps == (
        TaskStep(
            kind="tool",
            purpose="OpenClawを検索する",
            name="web_search_pages",
            arguments={"query": "openclaw"},
        ),
        TaskStep(
            kind="llm",
            purpose="検索結果から企業内活用案を提案する",
            name=None,
            arguments=None,
        ),
    )
    assert plan.limitations == ("社内規程への適合は未確認",)
    assert "tool: OpenClawを検索する" in (events[0].notes or "")
    assert "Webページの情報収集 (web_search_pages)" not in (events[0].notes or "")
    state = graph.conversation_state(123)
    assert [step["kind"] for step in state["plan_steps"]] == ["tool", "llm"]
    assert [step["name"] for step in state["plan_steps"]] == ["web_search_pages", None]
    assert [step["status"] for step in state["plan_steps"]] == [
        "completed",
        "completed",
    ]


def test_search_artifact_is_saved_to_conversation_context() -> None:
    artifact = {
        "type": "web_search_pages",
        "query": "生成AI",
        "search_results": [
            {
                "rank": 1,
                "title": "検索結果",
                "url": "https://example.test/article",
                "snippet": "概要",
            }
        ],
        "pages": [
            {
                "rank": 1,
                "url": "https://example.test/article",
                "final_url": "https://example.test/article",
                "title": "検索結果",
                "text": "保存される本文",
                "text_truncated": False,
                "fetch_ok": True,
                "error": None,
            }
        ],
    }
    graph = TicketConversationGraph(
        task_orchestrator=FakeOrchestrator(artifacts=(artifact,)),  # type: ignore[arg-type]
        llm=FakeLLM([]),
        checkpointer=InMemorySaver(),
        ai_user_id=42,
    )

    graph.run(issue=_issue())

    state = graph.conversation_state(123)
    assert state["artifacts"][0] == artifact
    assert any("保存される本文" in str(message.content) for message in state["messages"])
