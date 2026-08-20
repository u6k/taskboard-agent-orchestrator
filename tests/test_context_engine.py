from __future__ import annotations

import json
from typing import Any

import pytest
from langchain.tools import tool

from taskboard_agent.artifacts import InMemoryArtifactStore
from taskboard_agent.context_engine import (
    ContextEngine,
    ContextEngineError,
    ContextLimitExceeded,
    ConversationTurn,
    SessionCheckpoint,
    WorkingMemory,
)
from taskboard_agent.llm import LLMResponse
from taskboard_agent.structured_output import task_plan_response_format
from taskboard_agent.task_executor import (
    TaskPlan,
    TaskPlanningError,
    TaskStep,
    validate_task_plan,
    _repair_tool_step_arguments,
)


class FakeCompactionLLM:
    model = "test-model"

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {
            "summary": "案2を採用し、6セクション構成と設定を確定した。",
            "decisions": ["案2を採用"],
            "constraints": ["6セクション"],
            "open_questions": [],
            "current_position": "セクション2を執筆する",
            "selected_artifact_ids": [],
        }
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        self.calls.append((messages, kwargs))
        return LLMResponse(content=json.dumps(self.response, ensure_ascii=False))


def _memory() -> WorkingMemory:
    return WorkingMemory(
        issue={"id": 10, "subject": "小説を書く", "description": "案を検討する"},
        run_status="planned",
        active_artifacts={},
    )


def test_context_includes_only_selected_artifact_body() -> None:
    store = InMemoryArtifactStore()
    selected = store.put(
        {"section_2": "港町で対立が深まる"},
        kind="outline",
        label="section-outline",
    )
    unselected = store.put(
        {"section_6": "結末の本文"},
        kind="draft",
        label="section-6-draft",
    )
    engine = ContextEngine(
        llm=FakeCompactionLLM(),
        artifact_store=store,
        context_window_tokens=32768,
    )

    prepared = engine.prepare(
        working_memory=_memory(),
        checkpoint=SessionCheckpoint(),
        recent_turns=[ConversationTurn("journal-1", "user", "セクション2を書いて")],
        artifact_refs=[selected, unselected],
        selected_artifact_ids=(selected.artifact_id,),
    )

    prompt = json.dumps(prepared.messages, ensure_ascii=False)
    assert "港町で対立が深まる" in prompt
    assert "結末の本文" not in prompt
    assert selected.artifact_id in prompt
    assert unselected.artifact_id in prompt


def test_context_compacts_old_turns_and_preserves_recent_complete_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "taskboard_agent.context_engine.estimate_message_tokens",
        lambda _model, messages: len(json.dumps(messages, ensure_ascii=False)),
    )
    monkeypatch.setattr(
        "taskboard_agent.context_engine.estimate_text_tokens",
        lambda _model, text: len(text),
    )
    llm = FakeCompactionLLM()
    engine = ContextEngine(
        llm=llm,
        artifact_store=InMemoryArtifactStore(),
        context_window_tokens=16384,
    )
    turns = [
        ConversationTurn(f"turn-{index}", "user" if index % 2 else "assistant", "x" * 450)
        for index in range(1, 31)
    ]

    prepared = engine.prepare(
        working_memory=_memory(),
        checkpoint=SessionCheckpoint(summary="以前の要約"),
        recent_turns=turns,
        artifact_refs=[],
    )

    assert llm.calls[0][1]["operation"] == "session_compaction"
    assert prepared.checkpoint.summary.startswith("案2")
    assert prepared.checkpoint.compacted_through_turn_id is not None
    assert 0 < len(prepared.recent_turns) < len(turns)
    assert prepared.recent_turns[-1].id == "turn-30"


def test_context_rejects_invalid_compaction_artifact_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "taskboard_agent.context_engine.estimate_message_tokens",
        lambda _model, messages: len(json.dumps(messages, ensure_ascii=False)),
    )
    monkeypatch.setattr(
        "taskboard_agent.context_engine.estimate_text_tokens",
        lambda _model, text: len(text),
    )
    llm = FakeCompactionLLM(
        {
            "summary": "要約",
            "decisions": [],
            "constraints": [],
            "open_questions": [],
            "current_position": "継続中",
            "selected_artifact_ids": ["missing"],
        }
    )
    engine = ContextEngine(
        llm=llm,
        artifact_store=InMemoryArtifactStore(),
        context_window_tokens=16384,
    )
    previous = SessionCheckpoint(summary="以前のcheckpoint")

    with pytest.raises(ContextEngineError, match="unknown artifact"):
        engine.prepare(
            working_memory=_memory(),
            checkpoint=previous,
            recent_turns=[
                ConversationTurn(f"turn-{index}", "user", "x" * 700)
                for index in range(20)
            ],
            artifact_refs=[],
        )
    assert previous.summary == "以前のcheckpoint"


def test_context_reports_single_latest_turn_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "taskboard_agent.context_engine.estimate_message_tokens",
        lambda _model, messages: len(json.dumps(messages, ensure_ascii=False)),
    )
    monkeypatch.setattr(
        "taskboard_agent.context_engine.estimate_text_tokens",
        lambda _model, text: len(text),
    )
    engine = ContextEngine(
        llm=FakeCompactionLLM(),
        artifact_store=InMemoryArtifactStore(),
        context_window_tokens=16384,
    )

    with pytest.raises(ContextLimitExceeded) as captured:
        engine.prepare(
            working_memory=_memory(),
            checkpoint=SessionCheckpoint(),
            recent_turns=[ConversationTurn("latest", "user", "z" * 12000)],
            artifact_refs=[],
        )
    assert captured.value.breakdown["largest_turn_tokens"] == 12000


def test_task_plan_artifact_enum_and_dependency_validation() -> None:
    artifact_id = "a" * 64
    schema = task_plan_response_format(
        skill_names=[], tool_names=[], artifact_ids=[artifact_id]
    )["json_schema"]["schema"]
    item_schema = schema["properties"]["steps"]["items"]
    assert item_schema["properties"]["input_artifact_ids"]["items"]["enum"] == [
        artifact_id
    ]

    invalid = TaskPlan(
        decision="no_skill",
        reason="本文を書く",
        steps=(
            TaskStep(kind="llm", purpose="1", depends_on=(1,)),
        ),
    )
    with pytest.raises(TaskPlanningError, match="earlier steps"):
        validate_task_plan(invalid, skills=[], tools=[], artifact_ids={artifact_id})

    missing = TaskPlan(
        decision="no_skill",
        reason="本文を書く",
        steps=(
            TaskStep(kind="llm", purpose="1", input_artifact_ids=("missing",)),
        ),
    )
    with pytest.raises(TaskPlanningError, match="unknown artifacts"):
        validate_task_plan(missing, skills=[], tools=[], artifact_ids={artifact_id})

    duplicate = TaskPlan(
        decision="no_skill",
        reason="本文を書く",
        steps=(
            TaskStep(kind="llm", purpose="1"),
            TaskStep(kind="llm", purpose="2", depends_on=(1, 1)),
        ),
    )
    with pytest.raises(TaskPlanningError, match="duplicate"):
        validate_task_plan(duplicate, skills=[], tools=[], artifact_ids=set())

    future = TaskPlan(
        decision="no_skill",
        reason="本文を書く",
        steps=(
            TaskStep(kind="llm", purpose="1"),
            TaskStep(kind="llm", purpose="2", depends_on=(3,)),
            TaskStep(kind="llm", purpose="3"),
        ),
    )
    with pytest.raises(TaskPlanningError, match="earlier steps"):
        validate_task_plan(future, skills=[], tools=[], artifact_ids=set())


def test_fiction_section_two_uses_only_settings_outline_and_previous_draft() -> None:
    store = InMemoryArtifactStore()
    settings = store.put({"world": "浮遊都市", "relationships": "姉妹"}, kind="settings", label="settings")
    outline = store.put({"section": 2, "synopsis": "再会と対立"}, kind="outline", label="section-2-outline")
    previous = store.put({"section": 1, "text": "第一節本文"}, kind="draft", label="section-1-draft")
    unused = store.put({"ideas": ["案1", "案2"]}, kind="ideas", label="ideas")
    memory = WorkingMemory(
        issue={"id": 99, "subject": "小説", "description": "セクション2を執筆"},
        active_artifacts={
            "settings": {"version": 1, "artifact_id": settings.artifact_id},
            "section-2-outline": {"version": 1, "artifact_id": outline.artifact_id},
            "section-1-draft": {"version": 1, "artifact_id": previous.artifact_id},
        },
    )
    prepared = ContextEngine(
        llm=FakeCompactionLLM(),
        artifact_store=store,
        context_window_tokens=32768,
    ).prepare(
        working_memory=memory,
        checkpoint=SessionCheckpoint(summary="案2と6セクションを確定"),
        recent_turns=[ConversationTurn("journal-6", "user", "引き続きセクション2を書いて")],
        artifact_refs=[settings, outline, previous, unused],
        selected_artifact_ids=(settings.artifact_id, outline.artifact_id, previous.artifact_id),
    )

    prompt = json.dumps(prepared.messages, ensure_ascii=False)
    assert "浮遊都市" in prompt
    assert "再会と対立" in prompt
    assert "第一節本文" in prompt
    assert "案1" not in prompt


def test_tool_missing_argument_uses_exact_selected_artifact_field_and_rejects_conflict() -> None:
    @tool
    def publish(title: str) -> dict[str, Any]:
        """Publish a title."""
        return {"title": title}

    step = TaskStep(kind="tool", name="publish", purpose="公開する")
    plan = TaskPlan(decision="use_tools", reason="公開", tool_names=("publish",))
    arguments, events, conflict = _repair_tool_step_arguments(
        tool=publish,
        step=step,
        plan=plan,
        issue={"selected_artifacts": [{"content": {"title": "確定タイトル"}}]},
        step_context=[],
    )
    assert arguments == {"title": "確定タイトル"}
    assert events
    assert conflict is None

    arguments, _, conflict = _repair_tool_step_arguments(
        tool=publish,
        step=step,
        plan=plan,
        issue={
            "selected_artifacts": [
                {"content": {"title": "候補A"}},
                {"content": {"title": "候補B"}},
            ]
        },
        step_context=[],
    )
    assert arguments == {}
    assert conflict is not None
