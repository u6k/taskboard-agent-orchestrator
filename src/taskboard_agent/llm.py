from __future__ import annotations

import json
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any

import litellm
from langchain_core.callbacks import BaseCallbackHandler

from taskboard_agent.logging_config import current_trace_id


logger = logging.getLogger(__name__)


class CommentGenerationError(RuntimeError):
    """Raised when an updated issue description cannot be generated."""


class LLMError(RuntimeError):
    """Raised when the configured language model cannot be called safely."""


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: tuple[LLMToolCall, ...] = ()
    raw: Any | None = None


class LiteLLMClient:
    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._system_prompt = system_prompt.strip() if system_prompt else None

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_base(self) -> str | None:
        return self._api_base

    @property
    def timeout_seconds(self) -> int | None:
        return self._timeout_seconds

    def supports_function_calling(self) -> bool:
        try:
            return bool(litellm.supports_function_calling(model=self._model))
        except Exception as exc:  # pragma: no cover - provider metadata can vary.
            raise LLMError(f"failed to inspect model capabilities: {exc}") from exc

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        response_format: dict[str, Any] | None = None,
        operation: str = "completion",
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": with_agent_system_prompt(messages, self._system_prompt),
        }
        if self._api_base is not None:
            kwargs["base_url"] = self._api_base
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._timeout_seconds is not None:
            kwargs["timeout"] = self._timeout_seconds
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
            if response_format.get("type") == "json_schema":
                kwargs["enable_json_schema_validation"] = True
        metrics = _input_metrics(
            model=self._model,
            messages=kwargs["messages"],
            tools=tools,
        )
        started = time.perf_counter()
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:  # pragma: no cover - provider exceptions vary.
            _log_llm_metrics(
                operation=operation,
                model=self._model,
                request_id=None,
                input_metrics=metrics,
                duration_seconds=time.perf_counter() - started,
                output_chars=0,
                success=False,
                exception_type=type(exc).__name__,
            )
            raise LLMError(f"failed to call language model: {exc}") from exc
        llm_response = _to_llm_response(response)
        _log_llm_metrics(
            operation=operation,
            model=self._model,
            request_id=_response_request_id(response),
            input_metrics=metrics,
            duration_seconds=time.perf_counter() - started,
            output_chars=len(llm_response.content),
            success=True,
            exception_type=None,
        )
        return llm_response


def complete_with_operation(
    client: Any,
    messages: list[dict[str, Any]],
    *,
    operation: str,
    **kwargs: Any,
) -> Any:
    """Passes operation to instrumented clients while keeping simple test ports usable."""
    try:
        parameters = inspect.signature(client.complete).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    supports_operation = any(
        parameter.name == "operation"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if supports_operation:
        return client.complete(messages, operation=operation, **kwargs)
    return client.complete(messages, **kwargs)


def with_agent_system_prompt(
    messages: list[dict[str, Any]],
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    copied = [dict(message) for message in messages]
    if system_prompt is None or not system_prompt.strip():
        return copied
    agent_message = {
        "role": "system",
        "content": (
            "以下は担当エージェント固有の補助指示です。"
            "共通の業務制御、出力形式、tool policy、dry-run、承認規則と矛盾する場合は、"
            "共通規則を優先してください。\n\n"
            f"{system_prompt.strip()}"
        ),
    }
    insert_at = 0
    while insert_at < len(copied) and copied[insert_at].get("role") == "system":
        insert_at += 1
    copied.insert(insert_at, agent_message)
    return copied


class LiteLLMDescriptionGenerator:
    def __init__(self, llm: LiteLLMClient) -> None:
        self._llm = llm

    def generate(self, issue: dict[str, Any]) -> str:
        prompt = build_issue_prompt(issue)
        try:
            response = self._llm.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "あなたはRedmineチケットを読む業務AIです。"
                            "作業は実行せず、チケットのDescriptionを日本語で整理してください。"
                            "目的、作業内容、完了条件、課題を簡潔に整理し、推測は課題に含めてください。"
                            "ユーザーが記載した元文章は省略・改変せず、最後の見出しに残してください。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                operation="structured_final",
            )
        except Exception as exc:  # pragma: no cover - SDK exceptions vary by version.
            raise CommentGenerationError(f"failed to generate comment: {exc}") from exc

        comment = response.content
        if comment.strip() == "":
            raise CommentGenerationError("generated comment was empty")
        return comment.strip()


def build_issue_prompt(issue: dict[str, Any]) -> str:
    original_description = issue.get("description") or "(未記載)"
    fields = {
        "ID": issue.get("id"),
        "題名": issue.get("subject"),
        "作成者": _named_value(issue.get("author")),
        "担当者": _named_value(issue.get("assigned_to")),
        "ステータス": _named_value(issue.get("status")),
        "優先度": _named_value(issue.get("priority")),
        "プロジェクト": _named_value(issue.get("project")),
        "トラッカー": _named_value(issue.get("tracker")),
        "開始日": issue.get("start_date"),
        "期日": issue.get("due_date"),
    }
    field_text = "\n".join(f"- {key}: {value or '(未設定)'}" for key, value in fields.items())

    return (
        "次のRedmineチケットのDescriptionを更新する本文だけを出力してください。\n"
        "余計な前置き、コードブロック、説明文は出力しないでください。\n"
        "フォーマットは必ず次の形にしてください。\n\n"
        "# 目的\n"
        "{目的}\n\n"
        "# 実施すべき内容\n"
        "- {作業内容1}\n"
        "- {作業内容2}\n\n"
        "# 完了条件\n"
        "- {完了条件1}\n"
        "- {完了条件2}\n\n"
        "# 課題\n"
        "- {課題点、不明点、確認したい点}\n\n"
        "# ユーザーが記載した元文章\n"
        "{もともとDescriptionにユーザーが記載していた文章}\n\n"
        "元の説明文が空の場合は、最後の見出しの本文を「(未記載)」にしてください。\n\n"
        f"チケット情報:\n{field_text}\n\n"
        "ユーザーが記載した元Description:\n"
        "<<<ORIGINAL_DESCRIPTION\n"
        f"{original_description}\n"
        "ORIGINAL_DESCRIPTION"
    )


def _named_value(value: Any) -> str | None:
    if isinstance(value, dict):
        name = value.get("name")
        identifier = value.get("id")
        if name and identifier:
            return f"{name} (ID: {identifier})"
        if name:
            return str(name)
        if identifier:
            return f"ID: {identifier}"
    if value is None:
        return None
    return str(value)


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


def _to_llm_response(response: Any) -> LLMResponse:
    choices = _get(response, "choices", [])
    if not choices:
        return LLMResponse(content="", raw=response)
    first_choice = choices[0]
    message = _get(first_choice, "message", {})
    content = _get(message, "content", "") or ""
    tool_calls: list[LLMToolCall] = []
    for item in _get(message, "tool_calls", []) or []:
        function = _get(item, "function", {})
        name = _get(function, "name", "")
        arguments = _get(function, "arguments", "") or "{}"
        call_id = _get(item, "id", "")
        if isinstance(name, str) and name:
            tool_calls.append(
                LLMToolCall(
                    id=str(call_id or name),
                    name=name,
                    arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
                )
            )
    return LLMResponse(
        content=content if isinstance(content, str) else str(content),
        tool_calls=tuple(tool_calls),
        raw=response,
    )


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _input_metrics(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> dict[str, int]:
    serialized = json.dumps(messages, ensure_ascii=False, default=str)
    try:
        estimated_tokens = litellm.token_counter(model=model, messages=messages)
        if not isinstance(estimated_tokens, int) or estimated_tokens < 0:
            raise ValueError("invalid token count")
    except Exception:
        estimated_tokens = max(1, len(serialized) // 4)
    return {
        "message_count": len(messages),
        "input_chars": len(serialized),
        "input_bytes": len(serialized.encode("utf-8")),
        "estimated_tokens": estimated_tokens,
        "tool_count": len(tools or []),
    }


def _log_llm_metrics(
    *,
    operation: str,
    model: str,
    request_id: str | None,
    input_metrics: dict[str, int],
    duration_seconds: float,
    output_chars: int,
    success: bool,
    exception_type: str | None,
) -> None:
    payload: dict[str, Any] = {
        "event": "llm_call",
        "trace_id": current_trace_id(),
        "operation": operation,
        "model": model,
        "request_id": request_id,
        **input_metrics,
        "duration_seconds": round(duration_seconds, 6),
        "output_chars": output_chars,
        "success": success,
        "exception_type": exception_type,
    }
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _response_request_id(response: Any) -> str | None:
    value = _get(response, "id")
    return str(value) if value else None


class LLMCallMetricsCallback(BaseCallbackHandler):
    """Logs LangChain chat model calls without recording message bodies."""

    def __init__(self, *, model: str, operation: str, tool_count: int = 0) -> None:
        self._model = model
        self._operation = operation
        self._tool_count = tool_count
        self._runs: dict[str, tuple[float, dict[str, int]]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        flattened = messages[0] if messages else []
        message_dicts = [
            {
                "role": getattr(message, "type", "message"),
                "content": getattr(message, "content", ""),
            }
            for message in flattened
        ]
        metrics = _input_metrics(model=self._model, messages=message_dicts, tools=None)
        metrics["tool_count"] = self._tool_count
        self._runs[str(run_id)] = (
            time.perf_counter(),
            metrics,
        )

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        started, metrics = self._runs.pop(
            str(run_id), (time.perf_counter(), _empty_input_metrics())
        )
        output_chars = 0
        for generation_group in getattr(response, "generations", []) or []:
            for generation in generation_group or []:
                text = getattr(generation, "text", "") or ""
                message = getattr(generation, "message", None)
                content = getattr(message, "content", "") if message is not None else ""
                output_chars += len(str(content or text))
        _log_llm_metrics(
            operation=self._operation,
            model=self._model,
            request_id=str(run_id),
            input_metrics=metrics,
            duration_seconds=time.perf_counter() - started,
            output_chars=output_chars,
            success=True,
            exception_type=None,
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        started, metrics = self._runs.pop(
            str(run_id), (time.perf_counter(), _empty_input_metrics())
        )
        _log_llm_metrics(
            operation=self._operation,
            model=self._model,
            request_id=str(run_id),
            input_metrics=metrics,
            duration_seconds=time.perf_counter() - started,
            output_chars=0,
            success=False,
            exception_type=type(error).__name__,
        )


def _empty_input_metrics() -> dict[str, int]:
    return {
        "message_count": 0,
        "input_chars": 0,
        "input_bytes": 0,
        "estimated_tokens": 0,
        "tool_count": 0,
    }
