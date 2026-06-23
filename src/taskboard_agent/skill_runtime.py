from __future__ import annotations

import json
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from taskboard_agent.agent import AgentRunResult
from taskboard_agent.llm import LLMResponse
from taskboard_agent.skills import Skill
from taskboard_agent.structured_output import skill_execution_response_format
from taskboard_agent.tools import ToolRegistry


SkillEventKind = Literal["start", "progress", "final_review", "final_return"]
MAX_LLM_COMMENT_CHARS = 2000


class SkillRuntimeError(RuntimeError):
    """Raised when a skill run cannot be interpreted safely."""


class SkillAgentPort(Protocol):
    def run(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: ToolRegistry | None = None,
        allow_writes: bool = False,
        approved_tools: set[str] | None = None,
        on_llm_response: Callable[[LLMResponse], None] | None = None,
        response_format: dict[str, Any] | None = None,
        return_after_tool_names: set[str] | None = None,
    ) -> AgentRunResult:
        ...


@dataclass(frozen=True)
class SkillEvent:
    kind: SkillEventKind
    notes: str | None


SkillEventSink = Callable[[SkillEvent], None]


@dataclass(frozen=True)
class SkillExecutionResult:
    status: str
    events: tuple[SkillEvent, ...]
    target_url: str | None = None
    page_title: str | None = None
    briefing: str | None = None
    bookmark_url: str | None = None
    bookmark_payload: dict[str, Any] | None = None
    artifacts: tuple[dict[str, Any], ...] = ()
    dry_run: bool = False


@dataclass(frozen=True)
class ScriptedSkillContext:
    issue: dict[str, Any]
    task_input: dict[str, Any]
    dry_run: bool
    tools: ToolRegistry

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.tools.execute(
            name,
            arguments,
            allow_writes=not self.dry_run,
        ).content


class ScriptedSkillRunner:
    def __init__(self, *, skill: Skill, tools: ToolRegistry) -> None:
        self._skill = skill
        self._tools = tools

    def run(
        self,
        *,
        issue: dict[str, Any],
        task_input: dict[str, Any] | None = None,
        dry_run: bool = False,
        emit_event: SkillEventSink | None = None,
    ) -> SkillExecutionResult:
        del emit_event
        self._tools.require_registered(self._skill.required_tools)
        run = _load_scripted_skill(self._skill)
        result = run(
            ScriptedSkillContext(
                issue=issue,
                task_input=task_input or {},
                dry_run=dry_run,
                tools=self._tools,
            )
        )
        if not isinstance(result, SkillExecutionResult):
            raise SkillRuntimeError(
                f"skill runner must return SkillExecutionResult: {self._skill.name}"
            )
        return _add_agent_completion_notes(result)


class GenericSkillRunner:
    def __init__(
        self,
        *,
        skill: Skill,
        tools: ToolRegistry,
        skill_agent: SkillAgentPort,
    ) -> None:
        self._skill = skill
        self._tools = tools
        self._skill_agent = skill_agent

    def run(
        self,
        *,
        issue: dict[str, Any],
        task_input: dict[str, Any] | None = None,
        dry_run: bool = False,
        emit_event: SkillEventSink | None = None,
    ) -> SkillExecutionResult:
        self._tools.require_registered(self._skill.required_tools)
        llm_events: list[SkillEvent] = []

        def record_llm_response(response: LLMResponse) -> None:
            for event in llm_response_events(response):
                if emit_event is not None:
                    emit_event(event)
                else:
                    llm_events.append(event)

        agent_result = self._skill_agent.run(
            _build_skill_messages(
                issue=issue,
                skill=self._skill,
                task_input=task_input or {},
                dry_run=dry_run,
            ),
            tools=self._tools,
            allow_writes=not dry_run,
            on_llm_response=record_llm_response,
            response_format=skill_execution_response_format(),
        )
        final = _parse_final_json(agent_result.final_text)
        result = _to_skill_execution_result(
            final,
            skill_name=self._skill.name,
            dry_run=dry_run,
        )
        result = _with_tool_artifacts(result, agent_result.tool_results)
        if emit_event is not None or not llm_events:
            return result
        return _insert_after_start(result, tuple(llm_events))


def _build_skill_messages(
    *,
    issue: dict[str, Any],
    skill: Skill,
    task_input: dict[str, Any],
    dry_run: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "あなたはRedmineチケットを処理するAIエージェントです。"
                "必ず提供されたSKILL.mdの手順と利用可能なfunction toolsだけで作業してください。"
                "tool結果のok=falseは失敗として扱い、必要なら追加作業を止めてユーザー確認を求めてください。"
                "最終的な作業状態とRedmine向け報告を返してください。出力構造はAPI側で指定されています。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"スキル名: {skill.name}\n"
                f"スキル説明: {skill.description}\n\n"
                "SKILL.md:\n"
                f"{skill.body}\n\n"
                "実行コンテキスト:\n"
                f"{json.dumps({'issue': issue, 'task_input': task_input, 'dry_run': dry_run}, ensure_ascii=False)}\n\n"
                "notesにはRedmineへ投稿する自然な日本語の作業報告またはユーザーに求めることを記載し、"
                "実行内容、確認内容、成果物、未処理事項を含めてください。"
            ),
        },
    ]


def _to_skill_execution_result(
    final: dict[str, Any],
    *,
    skill_name: str,
    dry_run: bool,
) -> SkillExecutionResult:
    status = final.get("status")
    if status not in ("processed", "needs_user", "missing_tool", "failed", "already_done"):
        raise SkillRuntimeError("skill final JSON missing valid status")
    notes = final.get("notes")
    if not isinstance(notes, str) or notes.strip() == "":
        raise SkillRuntimeError("skill final JSON missing notes")

    events: list[SkillEvent] = [SkillEvent("start", "作業を開始します。")]
    if status == "processed":
        events.append(SkillEvent("progress", f"スキル `{skill_name}` を実行しました。"))
        events.append(SkillEvent("final_review", notes.strip()))
    elif status == "already_done":
        events.append(SkillEvent("progress", f"スキル `{skill_name}` を実行しました。"))
        events.append(SkillEvent("final_review", notes.strip()))
    elif status in ("needs_user", "missing_tool"):
        events.append(SkillEvent("final_review", notes.strip()))
    else:
        events.append(SkillEvent("final_return", notes.strip()))

    return SkillExecutionResult(
        status="dry_run" if dry_run and status == "processed" else status,
        events=tuple(events),
        target_url=_string_or_none(final.get("target_url")),
        page_title=_string_or_none(final.get("page_title")),
        briefing=_string_or_none(final.get("briefing")),
        bookmark_url=_string_or_none(final.get("bookmark_url")),
        bookmark_payload=_dict_or_none(final.get("bookmark_payload")),
        dry_run=dry_run,
    )


def _parse_final_json(output: str) -> dict[str, Any]:
    try:
        data = json.loads(_strip_json_fence(output))
    except json.JSONDecodeError as exc:
        raise SkillRuntimeError("skill final response was not valid JSON") from exc
    if not isinstance(data, dict):
        raise SkillRuntimeError("skill final response JSON was not an object")
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


def llm_response_events(response: LLMResponse) -> tuple[SkillEvent, ...]:
    parts: list[str] = []
    content = response.content.strip()
    if content:
        parts.append(f"エージェント出力:\n{_truncate(content)}")
    if response.tool_calls:
        lines = ["エージェントが次のtool呼び出しを判断しました。"]
        for tool_call in response.tool_calls:
            lines.append(
                f"- `{tool_call.name}` を呼び出します。引数: {_truncate(tool_call.arguments)}"
            )
        parts.append("\n".join(lines))
    if not parts:
        return ()
    return (SkillEvent("progress", "\n\n".join(parts)),)


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


def _with_tool_artifacts(
    result: SkillExecutionResult,
    tool_results: tuple[Any, ...],
) -> SkillExecutionResult:
    artifacts = _tool_artifacts(tool_results)
    if not artifacts:
        return result
    return SkillExecutionResult(
        status=result.status,
        events=tuple(_append_artifact_reports(result.events, artifacts)),
        target_url=result.target_url,
        page_title=result.page_title,
        briefing=result.briefing,
        bookmark_url=result.bookmark_url,
        bookmark_payload=result.bookmark_payload,
        artifacts=(*result.artifacts, *artifacts),
        dry_run=result.dry_run,
    )


def _tool_artifacts(tool_results: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    artifacts: list[dict[str, Any]] = []
    for result in tool_results:
        content = getattr(result, "content", None)
        if not isinstance(content, dict):
            continue
        artifact = content.get("context_artifact")
        if isinstance(artifact, dict):
            artifacts.append(artifact)
    return tuple(artifacts)


def _append_artifact_reports(
    events: tuple[SkillEvent, ...],
    artifacts: tuple[dict[str, Any], ...],
) -> list[SkillEvent]:
    report = _artifact_report(artifacts)
    if not report:
        return list(events)
    updated = list(events)
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].kind in ("final_review", "final_return"):
            notes = updated[index].notes or ""
            updated[index] = SkillEvent(
                updated[index].kind,
                f"{notes.rstrip()}\n\n{report}" if notes else report,
            )
            return updated
    updated.append(SkillEvent("final_review", report))
    return updated


def _artifact_report(artifacts: tuple[dict[str, Any], ...]) -> str:
    parts = [
        _web_search_pages_report(artifact)
        for artifact in artifacts
        if artifact.get("type") == "web_search_pages"
    ]
    return "\n\n".join(part for part in parts if part)


def _web_search_pages_report(artifact: dict[str, Any]) -> str:
    query = artifact.get("query")
    lines = [f"## 検索結果と本文取得結果\n\n検索キーワード: {query or '(未記録)'}"]
    search_results = artifact.get("search_results")
    if isinstance(search_results, list):
        lines.append("\n検索結果:")
        for item in search_results:
            if not isinstance(item, dict):
                continue
            rank = item.get("rank")
            title = item.get("title") or "(無題)"
            url = item.get("url") or ""
            snippet = item.get("snippet") or ""
            lines.append(f"- {rank}. {title} - {url}")
            if snippet:
                lines.append(f"  概要: {snippet}")
    pages = artifact.get("pages")
    if isinstance(pages, list):
        lines.append("\n本文取得:")
        for page in pages:
            if not isinstance(page, dict):
                continue
            rank = page.get("rank")
            if page.get("fetch_ok") is True:
                text = page.get("text")
                text_len = len(text) if isinstance(text, str) else 0
                final_url = page.get("final_url") or page.get("url") or ""
                title = page.get("title") or "(無題)"
                truncated = " / 切り詰めあり" if page.get("text_truncated") else ""
                lines.append(
                    f"- {rank}. 本文取得: 正常 / {title} / {text_len}字{truncated} / {final_url}"
                )
            else:
                url = page.get("url") or ""
                error = page.get("error") or "不明なエラー"
                lines.append(f"- {rank}. 本文取得: エラー / {url} / 理由: {error}")
    return "\n".join(lines)


def _truncate(value: str) -> str:
    if len(value) <= MAX_LLM_COMMENT_CHARS:
        return value
    return f"{value[:MAX_LLM_COMMENT_CHARS]}...(省略)"


def _load_scripted_skill(skill: Skill) -> Callable[[ScriptedSkillContext], Any]:
    if skill.runner is None:
        raise SkillRuntimeError(f"skill does not define a runner: {skill.name}")
    path = skill.path.parent / skill.runner
    module_name = f"_taskboard_agent_skill_runner_{skill.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SkillRuntimeError(f"failed to load skill runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise SkillRuntimeError(f"skill runner missing run(context): {path}")
    return run


def _add_agent_completion_notes(result: SkillExecutionResult) -> SkillExecutionResult:
    if result.status not in ("processed", "already_done", "dry_run"):
        return result
    events = list(result.events)
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event.kind == "final_review" and event.notes is None:
            events[index] = SkillEvent("final_review", "作業が終了しました。")
            break
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
