from __future__ import annotations

from typing import Any, Iterable


def task_plan_response_format(
    *,
    skill_names: Iterable[str],
    tool_names: Iterable[str],
    artifact_ids: Iterable[str] = (),
) -> dict[str, Any]:
    return _response_format(
        "task_plan",
        _task_plan_schema(
            skill_names=skill_names,
            tool_names=tool_names,
            artifact_ids=artifact_ids,
        ),
    )


def revision_plan_response_format(
    *,
    skill_names: Iterable[str],
    tool_names: Iterable[str],
    artifact_ids: Iterable[str] = (),
) -> dict[str, Any]:
    return _response_format(
        "revision_plan",
        {
            "type": "object",
            "properties": {
                "previous_work_summary": _non_empty_string_schema(),
                "feedback_summary": _non_empty_string_schema(),
                "requested_changes": _string_array_schema(),
                "keep_existing_results": _string_array_schema(),
                "work_to_redo": _string_array_schema(),
                "task_plan": _task_plan_schema(
                    skill_names=skill_names,
                    tool_names=tool_names,
                    artifact_ids=artifact_ids,
                ),
            },
            "required": [
                "previous_work_summary",
                "feedback_summary",
                "requested_changes",
                "keep_existing_results",
                "work_to_redo",
                "task_plan",
            ],
            "additionalProperties": False,
        },
    )


def generic_execution_response_format() -> dict[str, Any]:
    return _response_format(
        "generic_execution_result",
        {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["completed", "needs_user", "missing_tool"],
                },
                "notes": _non_empty_string_schema(),
            },
            "required": ["status", "notes"],
            "additionalProperties": False,
        },
    )


def tool_execution_response_format() -> dict[str, Any]:
    return _execution_response_format(
        "tool_execution_result",
        statuses=("processed", "needs_user", "missing_tool", "failed"),
    )


def skill_execution_response_format() -> dict[str, Any]:
    return _execution_response_format(
        "skill_execution_result",
        statuses=(
            "processed",
            "needs_user",
            "missing_tool",
            "failed",
            "already_done",
        ),
    )


def _execution_response_format(
    name: str,
    *,
    statuses: tuple[str, ...],
) -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    return _response_format(
        name,
        {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": list(statuses)},
                "notes": _non_empty_string_schema(),
                "target_url": nullable_string,
                "page_title": nullable_string,
                "briefing": nullable_string,
                "bookmark_url": nullable_string,
            },
            "required": [
                "status",
                "notes",
                "target_url",
                "page_title",
                "briefing",
                "bookmark_url",
            ],
            "additionalProperties": False,
        },
    )


def _task_plan_schema(
    *,
    skill_names: Iterable[str],
    tool_names: Iterable[str],
    artifact_ids: Iterable[str] = (),
) -> dict[str, Any]:
    skills = sorted({name for name in skill_names if name})
    tools = sorted({name for name in tool_names if name})
    step_names = sorted({*skills, *tools})
    artifacts = sorted({item for item in artifact_ids if item})
    return {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["use_skill", "use_tools", "no_skill", "needs_user"],
            },
            "reason": _non_empty_string_schema(),
            "skill_name": _nullable_enum_schema(skills),
            "tool_names": {
                "type": "array",
                "items": _enum_or_string_schema(tools),
            },
            "target_url": {"type": ["string", "null"]},
            "task_input": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "instruction": {"type": ["string", "null"]},
                            "target_url": {"type": ["string", "null"]},
                        },
                        "required": ["instruction", "target_url"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ]
            },
            "user_request": {"type": ["string", "null"]},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["skill", "tool", "llm", "unavailable"],
                        },
                        "name": _nullable_enum_schema(step_names),
                        "purpose": _non_empty_string_schema(),
                        "arguments": {
                            "anyOf": [
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "key": _non_empty_string_schema(),
                                            "value": {"type": "string"},
                                        },
                                        "required": ["key", "value"],
                                        "additionalProperties": False,
                                    },
                                },
                                {"type": "null"},
                            ]
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                            "uniqueItems": True,
                        },
                        "input_artifact_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": artifacts},
                            "uniqueItems": True,
                        },
                        "output_artifact_name": {"type": ["string", "null"]},
                    },
                    "required": [
                        "kind",
                        "name",
                        "purpose",
                        "arguments",
                        "depends_on",
                        "input_artifact_ids",
                        "output_artifact_name",
                    ],
                    "additionalProperties": False,
                },
            },
            "limitations": _string_array_schema(),
        },
        "required": [
            "decision",
            "reason",
            "skill_name",
            "tool_names",
            "target_url",
            "task_input",
            "user_request",
            "steps",
            "limitations",
        ],
        "additionalProperties": False,
    }


def _response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _string_array_schema() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _non_empty_string_schema() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _nullable_enum_schema(values: list[str]) -> dict[str, Any]:
    if not values:
        return {"type": "null"}
    return {"type": ["string", "null"], "enum": [*values, None]}


def _enum_or_string_schema(values: list[str]) -> dict[str, Any]:
    if not values:
        return {"type": "string"}
    return {"type": "string", "enum": values}
