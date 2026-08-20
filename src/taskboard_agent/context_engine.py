from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

import litellm

from taskboard_agent.artifacts import ArtifactRef, ArtifactStore
from taskboard_agent.llm import complete_with_operation


class ContextEngineError(RuntimeError):
    """Raised when model-visible context cannot be assembled safely."""


class ContextLimitExceeded(ContextEngineError):
    def __init__(self, breakdown: dict[str, int]) -> None:
        self.breakdown = breakdown
        super().__init__(
            "single input exceeds the model context budget: "
            + json.dumps(breakdown, ensure_ascii=False, sort_keys=True)
        )


class ContextLLMPort(Protocol):
    @property
    def model(self) -> str:
        ...

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        operation: str = "completion",
    ) -> Any:
        ...


@dataclass(frozen=True)
class ConversationTurn:
    id: str
    role: Literal["user", "assistant"]
    content: str
    journal_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["journal_ids"] = list(self.journal_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTurn:
        return cls(
            id=str(data.get("id") or "legacy-turn"),
            role="assistant" if data.get("role") == "assistant" else "user",
            content=str(data.get("content") or ""),
            journal_ids=tuple(
                item for item in data.get("journal_ids", []) if isinstance(item, int)
            ),
        )


@dataclass(frozen=True)
class WorkingMemory:
    issue: dict[str, Any]
    current_plan: dict[str, Any] | None = None
    plan_steps: tuple[dict[str, Any], ...] = ()
    run_status: str = "initialized"
    waiting_reason: str | None = None
    active_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "current_plan": self.current_plan,
            "plan_steps": list(self.plan_steps),
            "run_status": self.run_status,
            "waiting_reason": self.waiting_reason,
            "active_artifacts": self.active_artifacts,
        }


@dataclass(frozen=True)
class SessionCheckpoint:
    summary: str = ""
    decisions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    current_position: str = ""
    selected_artifact_ids: tuple[str, ...] = ()
    compacted_through_turn_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "decisions",
            "constraints",
            "open_questions",
            "selected_artifact_ids",
        ):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SessionCheckpoint:
        if not isinstance(data, dict):
            return cls()
        return cls(
            summary=str(data.get("summary") or ""),
            decisions=_string_tuple(data.get("decisions")),
            constraints=_string_tuple(data.get("constraints")),
            open_questions=_string_tuple(data.get("open_questions")),
            current_position=str(data.get("current_position") or ""),
            selected_artifact_ids=_string_tuple(data.get("selected_artifact_ids")),
            compacted_through_turn_id=(
                str(data["compacted_through_turn_id"])
                if data.get("compacted_through_turn_id")
                else None
            ),
        )


@dataclass(frozen=True)
class PreparedContext:
    messages: tuple[dict[str, Any], ...]
    checkpoint: SessionCheckpoint
    recent_turns: tuple[ConversationTurn, ...]
    estimated_tokens: int


class ContextEngine:
    def __init__(
        self,
        *,
        llm: ContextLLMPort,
        artifact_store: ArtifactStore,
        context_window_tokens: int,
    ) -> None:
        if context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        self._llm = llm
        self._model = str(getattr(llm, "model", "test-model"))
        self._artifact_store = artifact_store
        self.context_window_tokens = context_window_tokens

    @property
    def input_budget_tokens(self) -> int:
        output_reserve = max(4096, int(self.context_window_tokens * 0.15))
        safety_buffer = max(2048, int(self.context_window_tokens * 0.10))
        return max(1, self.context_window_tokens - output_reserve - safety_buffer)

    def prepare(
        self,
        *,
        working_memory: WorkingMemory,
        checkpoint: SessionCheckpoint,
        recent_turns: list[ConversationTurn],
        artifact_refs: list[ArtifactRef],
        selected_artifact_ids: tuple[str, ...] = (),
    ) -> PreparedContext:
        refs_by_id = {ref.artifact_id: ref for ref in artifact_refs}
        unknown = [item for item in selected_artifact_ids if item not in refs_by_id]
        if unknown:
            raise ContextEngineError(
                f"selected artifacts do not exist in this session: {', '.join(unknown)}"
            )
        selected = [
            (refs_by_id[artifact_id], self._artifact_store.get(artifact_id))
            for artifact_id in selected_artifact_ids
        ]
        fixed_messages = _context_messages(
            working_memory=working_memory,
            checkpoint=checkpoint,
            recent_turns=[],
            artifact_refs=artifact_refs,
            selected_artifacts=selected,
        )
        fixed_tokens = estimate_message_tokens(self._model, fixed_messages)
        single_turn_tokens = max(
            (estimate_text_tokens(self._model, turn.content) for turn in recent_turns),
            default=0,
        )
        single_artifact_tokens = max(
            (
                estimate_text_tokens(
                    self._model,
                    json.dumps(content, ensure_ascii=False, default=str),
                )
                for _, content in selected
            ),
            default=0,
        )
        if (
            single_turn_tokens > self.input_budget_tokens
            or single_artifact_tokens > self.input_budget_tokens
        ):
            raise ContextLimitExceeded(
                {
                    "input_budget_tokens": self.input_budget_tokens,
                    "fixed_context_tokens": fixed_tokens,
                    "largest_turn_tokens": single_turn_tokens,
                    "largest_artifact_tokens": single_artifact_tokens,
                }
            )

        messages = _context_messages(
            working_memory=working_memory,
            checkpoint=checkpoint,
            recent_turns=recent_turns,
            artifact_refs=artifact_refs,
            selected_artifacts=selected,
        )
        total = estimate_message_tokens(self._model, messages)
        if total <= self.input_budget_tokens:
            return PreparedContext(tuple(messages), checkpoint, tuple(recent_turns), total)

        compacted_checkpoint, kept_turns = self._compact(
            checkpoint=checkpoint,
            recent_turns=recent_turns,
            valid_artifact_ids=set(refs_by_id),
        )
        messages = _context_messages(
            working_memory=working_memory,
            checkpoint=compacted_checkpoint,
            recent_turns=kept_turns,
            artifact_refs=artifact_refs,
            selected_artifacts=selected,
        )
        total = estimate_message_tokens(self._model, messages)
        if total > self.input_budget_tokens:
            raise ContextLimitExceeded(
                {
                    "input_budget_tokens": self.input_budget_tokens,
                    "assembled_tokens": total,
                    "largest_turn_tokens": single_turn_tokens,
                    "largest_artifact_tokens": single_artifact_tokens,
                }
            )
        return PreparedContext(
            tuple(messages), compacted_checkpoint, tuple(kept_turns), total
        )

    def _compact(
        self,
        *,
        checkpoint: SessionCheckpoint,
        recent_turns: list[ConversationTurn],
        valid_artifact_ids: set[str],
    ) -> tuple[SessionCheckpoint, list[ConversationTurn]]:
        recent_budget = max(1, int(self.context_window_tokens * 0.25))
        kept_reversed: list[ConversationTurn] = []
        used = 0
        for turn in reversed(recent_turns):
            turn_tokens = estimate_text_tokens(self._model, turn.content)
            if kept_reversed and used + turn_tokens > recent_budget:
                break
            kept_reversed.append(turn)
            used += turn_tokens
        kept = list(reversed(kept_reversed))
        compact_count = len(recent_turns) - len(kept)
        if compact_count <= 0:
            raise ContextEngineError("context exceeds budget but no complete old turn can be compacted")
        old_turns = recent_turns[:compact_count]
        response = complete_with_operation(
            self._llm,
            [
                {
                    "role": "system",
                    "content": (
                        "Redmineチケット会話の古いturnを、後続作業で参照できるcheckpointへ圧縮してください。"
                        "事実、決定、制約、未解決事項、現在位置を保持し、存在しないartifact IDを作らないでください。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "previous_checkpoint": checkpoint.to_dict(),
                            "turns_to_compact": [turn.to_dict() for turn in old_turns],
                            "valid_artifact_ids": sorted(valid_artifact_ids),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format=session_checkpoint_response_format(),
            operation="session_compaction",
        )
        try:
            data = json.loads(str(response.content).strip())
            candidate = _validated_checkpoint(data, valid_artifact_ids)
        except (TypeError, ValueError, json.JSONDecodeError, KeyError) as exc:
            raise ContextEngineError(f"session compaction returned invalid data: {exc}") from exc
        candidate = SessionCheckpoint(
            **{
                **candidate.__dict__,
                "compacted_through_turn_id": old_turns[-1].id,
            }
        )
        return candidate, kept


def session_checkpoint_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "session_checkpoint",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "minLength": 1},
                    "decisions": {"type": "array", "items": {"type": "string"}},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "open_questions": {"type": "array", "items": {"type": "string"}},
                    "current_position": {"type": "string"},
                    "selected_artifact_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "summary",
                    "decisions",
                    "constraints",
                    "open_questions",
                    "current_position",
                    "selected_artifact_ids",
                ],
                "additionalProperties": False,
            },
        },
    }


def estimate_message_tokens(model: str, messages: list[dict[str, Any]]) -> int:
    try:
        value = litellm.token_counter(model=model, messages=messages)
        if isinstance(value, int) and value > 0:
            return value
    except Exception:
        pass
    return max(1, len(json.dumps(messages, ensure_ascii=False, default=str)) // 4)


def estimate_text_tokens(model: str, text: str) -> int:
    return estimate_message_tokens(model, [{"role": "user", "content": text}])


def _context_messages(
    *,
    working_memory: WorkingMemory,
    checkpoint: SessionCheckpoint,
    recent_turns: list[ConversationTurn],
    artifact_refs: list[ArtifactRef],
    selected_artifacts: list[tuple[ArtifactRef, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "以下はチケットsessionのmodel-visible contextです。"
                "Redmine progressログではなく、checkpoint、直近turn、選択artifactを根拠に作業してください。"
                "成果物参照の候補が複数あり一意に決められない場合は推測せず確認してください。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "working_memory": working_memory.to_dict(),
                    "session_checkpoint": checkpoint.to_dict(),
                    "artifact_catalog": [ref.to_dict() for ref in artifact_refs],
                },
                ensure_ascii=False,
                default=str,
            ),
        },
    ]
    messages.extend(
        {"role": turn.role, "content": turn.content} for turn in recent_turns
    )
    if selected_artifacts:
        messages.append(
            {
                "role": "user",
                "content": "選択された成果物本文:\n"
                + json.dumps(
                    [
                        {"artifact": ref.to_dict(), "content": content}
                        for ref, content in selected_artifacts
                    ],
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )
    return messages


def _validated_checkpoint(
    data: Any, valid_artifact_ids: set[str]
) -> SessionCheckpoint:
    if not isinstance(data, dict) or not isinstance(data.get("summary"), str):
        raise ValueError("summary is missing")
    if not data["summary"].strip():
        raise ValueError("summary is empty")
    selected = _string_tuple(data.get("selected_artifact_ids"))
    unknown = set(selected) - valid_artifact_ids
    if unknown:
        raise ValueError(f"unknown artifact ids: {', '.join(sorted(unknown))}")
    return SessionCheckpoint(
        summary=data["summary"].strip(),
        decisions=_string_tuple(data.get("decisions")),
        constraints=_string_tuple(data.get("constraints")),
        open_questions=_string_tuple(data.get("open_questions")),
        current_position=str(data.get("current_position") or ""),
        selected_artifact_ids=selected,
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        if value in (None, ()):
            return ()
        raise ValueError("expected a string array")
    return tuple(item.strip() for item in value if item.strip())
