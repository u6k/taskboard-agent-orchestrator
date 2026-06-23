from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Protocol

from taskboard_agent.llm import LLMResponse, LLMToolCall
from taskboard_agent.tools import (
    ToolExecutionResult,
    ToolRegistry,
    ToolRegistryError,
    parse_tool_arguments,
)


class AgentLLMPort(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        ...


@dataclass(frozen=True)
class AgentRunResult:
    final_text: str
    messages: tuple[dict[str, Any], ...]
    tool_results: tuple[ToolExecutionResult, ...]
    stopped_reason: str


class FunctionCallingAgent:
    def __init__(
        self,
        *,
        llm: AgentLLMPort,
        tools: ToolRegistry | None = None,
        max_steps: int = 8,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._max_steps = max_steps

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
        registry = tools or self._tools
        if registry is None:
            raise ToolRegistryError("function calling agent requires a tool registry")

        working_messages = list(messages)
        tool_results: list[ToolExecutionResult] = []

        for _step in range(self._max_steps):
            response = self._llm.complete(
                working_messages,
                tools=registry.litellm_tools(),
                tool_choice="auto",
            )
            if not response.tool_calls:
                if response.content:
                    working_messages.append(
                        {"role": "assistant", "content": response.content}
                    )
                final_response = response
                if response_format is not None:
                    final_response = self._llm.complete(
                        [
                            *working_messages,
                            {
                                "role": "user",
                                "content": (
                                    "これまでの会話とtool実行結果を基に、最終結果だけを"
                                    "指定された出力構造で返してください。追加のtoolは呼び出さないでください。"
                                ),
                            },
                        ],
                        response_format=response_format,
                    )
                    if final_response.content:
                        working_messages.append(
                            {"role": "assistant", "content": final_response.content}
                        )
                return AgentRunResult(
                    final_text=final_response.content,
                    messages=tuple(working_messages),
                    tool_results=tuple(tool_results),
                    stopped_reason="final",
                )

            if on_llm_response is not None:
                on_llm_response(response)
            working_messages.append(_assistant_tool_call_message(response))
            for tool_call in response.tool_calls:
                result = self._execute_tool_call(
                    tool_call,
                    tools=registry,
                    allow_writes=allow_writes,
                    approved_tools=approved_tools,
                )
                tool_results.append(result)
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result.to_json(),
                    }
                )
            if return_after_tool_names and any(
                result.name in return_after_tool_names for result in tool_results
            ):
                return AgentRunResult(
                    final_text="",
                    messages=tuple(working_messages),
                    tool_results=tuple(tool_results),
                    stopped_reason="tool_result",
                )

        return AgentRunResult(
            final_text="",
            messages=tuple(working_messages),
            tool_results=tuple(tool_results),
            stopped_reason="max_steps",
        )

    def _execute_tool_call(
        self,
        tool_call: LLMToolCall,
        *,
        tools: ToolRegistry,
        allow_writes: bool,
        approved_tools: set[str] | None,
    ) -> ToolExecutionResult:
        arguments = parse_tool_arguments(tool_call.name, tool_call.arguments)
        return tools.execute(
            tool_call.name,
            arguments,
            allow_writes=allow_writes,
            approved_tools=approved_tools,
        )


def _assistant_tool_call_message(response: LLMResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content or None,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            }
            for tool_call in response.tool_calls
        ],
    }
