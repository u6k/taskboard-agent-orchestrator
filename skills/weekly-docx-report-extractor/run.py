from __future__ import annotations

from typing import Any

from taskboard_agent.skill_runtime import (
    ScriptedSkillContext,
    SkillEvent,
    SkillExecutionResult,
)


def run(context: ScriptedSkillContext) -> SkillExecutionResult:
    attachments = _docx_attachments(context.issue.get("attachments"))
    if not attachments:
        return SkillExecutionResult(
            status="dry_run" if context.dry_run else "needs_user",
            events=(
                SkillEvent(
                    "final_review",
                    "処理対象のDOCX添付がありません。週報DOCXをチケットへ添付してください。",
                ),
            ),
            dry_run=context.dry_run,
        )

    events: list[SkillEvent] = []
    results: list[tuple[str, bool]] = []
    for attachment in attachments:
        filename = attachment["filename"]
        content_url = attachment.get("content_url")
        if not isinstance(content_url, str) or not content_url:
            events.append(
                SkillEvent(
                    "progress",
                    _failure_comment(filename, "添付情報の確認", "content_urlがありません。"),
                )
            )
            results.append((filename, False))
            continue

        extracted, error = _call(
            context,
            "extract_redmine_docx",
            {"filename": filename, "content_url": content_url},
        )
        if error is not None:
            events.append(
                SkillEvent("progress", _failure_comment(filename, "DOCX本文の抽出", error))
            )
            results.append((filename, False))
            continue
        text = extracted.get("text")
        if not isinstance(text, str) or not text:
            events.append(
                SkillEvent(
                    "progress",
                    _failure_comment(filename, "DOCX本文の抽出", "抽出本文が返されませんでした。"),
                )
            )
            results.append((filename, False))
            continue

        summarized, error = _call(
            context,
            "summarize_weekly_docx",
            {"filename": filename, "text": text},
        )
        if error is not None:
            events.append(
                SkillEvent("progress", _failure_comment(filename, "週報サマリーの生成", error))
            )
            results.append((filename, False))
            continue
        summary = summarized.get("summary")
        if not isinstance(summary, str) or not summary:
            events.append(
                SkillEvent(
                    "progress",
                    _failure_comment(filename, "週報サマリーの生成", "サマリーが返されませんでした。"),
                )
            )
            results.append((filename, False))
            continue
        events.append(SkillEvent("progress", summary))
        results.append((filename, True))

    success_count = sum(success for _, success in results)
    failure_count = len(results) - success_count
    events.append(SkillEvent("final_review", _completion_comment(results)))
    if context.dry_run:
        status = "dry_run"
    elif failure_count:
        status = "needs_user"
    else:
        status = "processed"
    return SkillExecutionResult(status=status, events=tuple(events), dry_run=context.dry_run)


def _docx_attachments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    attachments: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if isinstance(filename, str) and filename.lower().endswith(".docx"):
            attachments.append(item)
    return attachments


def _call(
    context: ScriptedSkillContext,
    name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    try:
        result = context.execute_tool(name, arguments)
    except Exception as exc:
        return {}, str(exc)
    if result.get("ok") is not True:
        error = result.get("error")
        return result, error if isinstance(error, str) and error else "toolの実行に失敗しました。"
    return result, None


def _failure_comment(filename: str, step: str, error: str) -> str:
    return (
        f"対象ファイル: {filename}\n\n"
        f"{step}に失敗しました。\n\n"
        f"- 理由: {error}"
    )


def _completion_comment(results: list[tuple[str, bool]]) -> str:
    success_count = sum(success for _, success in results)
    failure_count = len(results) - success_count
    lines = [
        "週報DOCXの全ファイル処理が完了しました。",
        "",
        f"- 対象: {len(results)}件",
        f"- 成功: {success_count}件",
        f"- 失敗: {failure_count}件",
        "- 処理ファイル:",
    ]
    lines.extend(
        f"  - {'成功' if success else '失敗'}: {filename}" for filename, success in results
    )
    return "\n".join(lines)
