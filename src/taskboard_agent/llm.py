from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import litellm


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
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

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
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
            if response_format.get("type") == "json_schema":
                kwargs["enable_json_schema_validation"] = True
        logger.debug(
            "LLM入力プロンプト model=%s payload=%s",
            self._model,
            _log_json(
                {
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": tool_choice if tools is not None else None,
                    "response_format": response_format,
                }
            ),
        )
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:  # pragma: no cover - provider exceptions vary.
            raise LLMError(f"failed to call language model: {exc}") from exc
        llm_response = _to_llm_response(response)
        logger.debug(
            "LLM出力内容 model=%s payload=%s",
            self._model,
            _log_json(
                {
                    "content": llm_response.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                        for tool_call in llm_response.tool_calls
                    ],
                }
            ),
        )
        return llm_response


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
                ]
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


def _log_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
