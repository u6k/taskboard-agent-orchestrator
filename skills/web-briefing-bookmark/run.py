from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from taskboard_agent.skill_runtime import (
    ScriptedSkillContext,
    SkillEvent,
    SkillExecutionResult,
)


# The authoritative workflow specification is documented in SKILL.md.
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")
URL_TRAILING_PUNCTUATION = ".,;:!?、。）」』】]}"


def run(context: ScriptedSkillContext) -> SkillExecutionResult:
    issue_id = context.issue.get("id")
    if not isinstance(issue_id, int):
        return _result(context, status="failed")

    intended_comments: list[str] = []
    target_url, url_error = _target_url(context)
    if target_url is None:
        notes = url_error or "対象URLを特定できませんでした。"
        if not _add_comment(context, issue_id, notes, intended_comments):
            return _result(context, status="failed", comments=intended_comments)
        return _result(
            context,
            status="needs_user",
            comments=intended_comments,
        )

    found, error = _call(context, "linkace_find_link", {"url": target_url})
    if error is not None:
        return _fail(
            context,
            issue_id,
            intended_comments,
            step="LinkAceの既存登録確認",
            error=error,
            target_url=target_url,
        )

    bookmark = found.get("bookmark")
    if found.get("found") is True and isinstance(bookmark, dict):
        if bookmark.get("has_source_list") is False:
            bookmark_url = _string(bookmark.get("web_url"))
            notes = (
                "ブックマークは登録済みです。リストID 1に属していないため、"
                "要約、登録、更新は行いません。"
            )
            if bookmark_url:
                notes += f"\n\n- 既存ブックマーク: {bookmark_url}"
            if not _add_comment(context, issue_id, notes, intended_comments):
                return _result(
                    context,
                    status="failed",
                    target_url=target_url,
                    bookmark_url=bookmark_url,
                    comments=intended_comments,
                )
            return _result(
                context,
                status="already_done",
                target_url=target_url,
                bookmark_url=bookmark_url,
                comments=intended_comments,
            )

    page, error = _call(context, "fetch_web_page", {"url": target_url})
    if error is not None:
        return _fail(
            context,
            issue_id,
            intended_comments,
            step="Webページの取得",
            error=error,
            target_url=target_url,
        )
    final_url = _string(page.get("url")) or target_url
    title = _string(page.get("title"))
    text = _string(page.get("text"))
    if not title or not text:
        return _fail(
            context,
            issue_id,
            intended_comments,
            step="Webページの取得",
            error="ページタイトルまたは本文が返されませんでした。",
            target_url=final_url,
            page_title=title,
        )
    fetch_notes = f"Webページを取得しました。\n\n- URL: {final_url}\n- タイトル: {title}"
    if not _add_comment(context, issue_id, fetch_notes, intended_comments):
        return _result(
            context,
            status="failed",
            target_url=final_url,
            page_title=title,
            comments=intended_comments,
        )

    summary, error = _call(
        context,
        "summarize_briefing",
        {"url": final_url, "title": title, "text": text},
    )
    if error is not None:
        return _fail(
            context,
            issue_id,
            intended_comments,
            step="ブリーフィング要約の生成",
            error=error,
            target_url=final_url,
            page_title=title,
        )
    briefing = _string(summary.get("briefing"))
    if not briefing:
        return _fail(
            context,
            issue_id,
            intended_comments,
            step="ブリーフィング要約の生成",
            error="要約本文が返されませんでした。",
            target_url=final_url,
            page_title=title,
        )
    summary_notes = f"ブリーフィング要約を取得しました。\n\n{briefing}"
    if not _add_comment(context, issue_id, summary_notes, intended_comments):
        return _result(
            context,
            status="failed",
            target_url=final_url,
            page_title=title,
            briefing=briefing,
            comments=intended_comments,
        )

    added, error = _call(
        context,
        "linkace_add_link",
        {"url": final_url, "title": title, "description": briefing},
    )
    if error is not None:
        return _fail(
            context,
            issue_id,
            intended_comments,
            step="LinkAceへのブックマーク登録",
            error=error,
            target_url=final_url,
            page_title=title,
            briefing=briefing,
            bookmark_payload=_dict(added.get("payload")),
        )

    payload = _dict(added.get("payload"))
    bookmark_result = _dict(added.get("bookmark"))
    bookmark_url = _string(bookmark_result.get("url"))
    action = _string(bookmark_result.get("action"))
    if context.dry_run:
        registration_notes = "LinkAceへブックマークを登録または更新する予定です。"
        status = "dry_run"
    elif action == "created":
        registration_notes = "LinkAceへブックマークを登録しました。"
        status = "processed"
    elif action == "updated":
        registration_notes = "LinkAceのブックマークを更新しました。"
        status = "processed"
    elif action == "already_exists":
        registration_notes = "ブックマークは登録済みのため、登録や更新は行いませんでした。"
        status = "already_done"
    else:
        return _fail(
            context,
            issue_id,
            intended_comments,
            step="LinkAceへのブックマーク登録",
            error="登録結果のactionを判定できませんでした。",
            target_url=final_url,
            page_title=title,
            briefing=briefing,
            bookmark_payload=payload,
        )
    if bookmark_url:
        registration_notes += f"\n\n- ブックマーク: {bookmark_url}"
    if not _add_comment(context, issue_id, registration_notes, intended_comments):
        status = "failed"

    return _result(
        context,
        status=status,
        target_url=final_url,
        page_title=title,
        briefing=briefing,
        bookmark_url=bookmark_url,
        bookmark_payload=payload,
        comments=intended_comments,
    )


def _target_url(context: ScriptedSkillContext) -> tuple[str | None, str | None]:
    explicit = context.task_input.get("target_url")
    if isinstance(explicit, str) and _is_http_url(explicit.strip()):
        return explicit.strip(), None

    urls: list[str] = []
    for key in ("description", "subject"):
        value = context.issue.get(key)
        if not isinstance(value, str):
            continue
        for match in URL_PATTERN.findall(value):
            url = match.rstrip(URL_TRAILING_PUNCTUATION)
            if _is_http_url(url) and url not in urls:
                urls.append(url)
    if len(urls) == 1:
        return urls[0], None
    if not urls:
        return None, "対象URLを特定できませんでした。チケットにHTTP(S) URLを1件追記してください。"
    return None, "対象URLを一意に特定できませんでした。処理するURLを1件だけ指定してください。"


def _is_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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
        return result, _string(result.get("error")) or "toolの実行に失敗しました。"
    return result, None


def _add_comment(
    context: ScriptedSkillContext,
    issue_id: int,
    notes: str,
    intended_comments: list[str],
) -> bool:
    intended_comments.append(notes)
    _, error = _call(
        context,
        "redmine_add_comment",
        {"issue_id": issue_id, "notes": notes},
    )
    return error is None


def _fail(
    context: ScriptedSkillContext,
    issue_id: int,
    intended_comments: list[str],
    *,
    step: str,
    error: str,
    target_url: str | None = None,
    page_title: str | None = None,
    briefing: str | None = None,
    bookmark_payload: dict[str, Any] | None = None,
) -> SkillExecutionResult:
    notes = f"{step}に失敗したため、後続処理を停止します。\n理由: {error}"
    _add_comment(context, issue_id, notes, intended_comments)
    return _result(
        context,
        status="failed",
        target_url=target_url,
        page_title=page_title,
        briefing=briefing,
        bookmark_payload=bookmark_payload,
        comments=intended_comments,
    )


def _result(
    context: ScriptedSkillContext,
    *,
    status: str,
    target_url: str | None = None,
    page_title: str | None = None,
    briefing: str | None = None,
    bookmark_url: str | None = None,
    bookmark_payload: dict[str, Any] | None = None,
    comments: list[str] | None = None,
) -> SkillExecutionResult:
    events: list[SkillEvent] = []
    if context.dry_run:
        events.extend(SkillEvent("progress", notes) for notes in comments or [])
    events.append(SkillEvent("final_review", None))
    return SkillExecutionResult(
        status=status,
        events=tuple(events),
        target_url=target_url,
        page_title=page_title,
        briefing=briefing,
        bookmark_url=bookmark_url,
        bookmark_payload=bookmark_payload,
        dry_run=context.dry_run,
    )


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
