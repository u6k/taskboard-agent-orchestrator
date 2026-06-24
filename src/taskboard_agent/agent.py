from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from taskboard_agent.llm import LLMResponse, LLMToolCall
from taskboard_agent.tools import ToolExecutionResult, require_tool_policy


@dataclass(frozen=True)
class AgentRunResult:
    final_text: str
    messages: tuple[Any, ...]
    tool_results: tuple[ToolExecutionResult, ...]
    stopped_reason: str


class LangChainAgentRunner:
    def __init__(
        self,
        *,
        model: Any,
        max_steps: int = 8,
    ) -> None:
        self._model = model
        self._max_steps = max_steps

    def run(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[BaseTool] | None = None,
        allow_writes: bool = False,
        approved_tools: set[str] | None = None,
        on_llm_response: Callable[[LLMResponse], None] | None = None,
        response_format: dict[str, Any] | None = None,
        return_after_tool_names: set[str] | None = None,
    ) -> AgentRunResult:
        if tools is None:
            raise RuntimeError("LangChain agent runner requires tools")
        guarded_tools = [
            _policy_guarded_tool(
                tool,
                allow_writes=allow_writes,
                approved_tools=approved_tools,
            )
            for tool in tools
        ]
        agent = create_agent(
            model=self._model,
            tools=guarded_tools,
            response_format=_langchain_response_schema(response_format),
            interrupt_after=["tools"] if return_after_tool_names else None,
        )
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": max(self._max_steps * 2, 2)},
        )
        output_messages = tuple(result.get("messages", ()))
        if on_llm_response is not None:
            for message in output_messages:
                response = _llm_response_from_message(message)
                if response.tool_calls or response.content:
                    on_llm_response(response)
        tool_results = tuple(_tool_results(output_messages))
        schema = _langchain_response_schema(response_format)
        structured_response = result.get("structured_response")
        if (
            structured_response is None
            and schema is not None
            and not return_after_tool_names
        ):
            structured_response = self._model.with_structured_output(schema).invoke(
                [
                    *output_messages,
                    HumanMessage(
                        content=(
                            "これまでの会話とtool実行結果を基に、最終結果だけを"
                            "指定された出力構造で返してください。追加のtoolは呼び出さないでください。"
                        )
                    ),
                ]
            )
        final_text = (
            json.dumps(_jsonable_structured_response(structured_response), ensure_ascii=False)
            if structured_response is not None
            else _last_ai_content(output_messages)
        )
        stopped_reason = (
            "tool_result"
            if return_after_tool_names
            and any(item.name in return_after_tool_names for item in tool_results)
            else "final"
        )
        return AgentRunResult(
            final_text=final_text,
            messages=output_messages,
            tool_results=tool_results,
            stopped_reason=stopped_reason,
        )


def _policy_guarded_tool(
    original: BaseTool,
    *,
    allow_writes: bool,
    approved_tools: set[str] | None,
) -> BaseTool:
    def invoke_original(**kwargs: Any) -> Any:
        require_tool_policy(
            original,
            allow_writes=allow_writes,
            approved_tools=approved_tools,
        )
        return original.invoke(kwargs)

    return StructuredTool.from_function(
        invoke_original,
        name=original.name,
        description=original.description,
        args_schema=original.args_schema,
        infer_schema=False,
        response_format=original.response_format,
        extras=dict(original.extras or {}),
    )


def _langchain_response_schema(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if response_format is None:
        return None
    if response_format.get("type") != "json_schema":
        return response_format
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return response_format
    schema = dict(json_schema.get("schema") or {})
    name = json_schema.get("name")
    if isinstance(name, str) and "title" not in schema:
        schema["title"] = name
    return schema


def _jsonable_structured_response(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _tool_results(messages: tuple[Any, ...]) -> list[ToolExecutionResult]:
    results: list[ToolExecutionResult] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = _message_content_text(message)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"result": content}
        if not isinstance(data, dict):
            data = {"result": data}
        results.append(ToolExecutionResult(name=message.name or "", content=data))
    return results


def _last_ai_content(messages: tuple[Any, ...]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return _message_content_text(message)
    return ""


def _llm_response_from_message(message: Any) -> LLMResponse:
    if not isinstance(message, AIMessage):
        return LLMResponse(content="")
    return LLMResponse(
        content=_message_content_text(message),
        tool_calls=tuple(
            LLMToolCall(
                id=str(tool_call.get("id") or tool_call.get("name") or ""),
                name=str(tool_call.get("name") or ""),
                arguments=json.dumps(tool_call.get("args") or {}, ensure_ascii=False),
            )
            for tool_call in message.tool_calls
            if tool_call.get("name")
        ),
    )


def _message_content_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)
