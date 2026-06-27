from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from taskboard_agent.skill_runtime import (
    ScriptedSkillContext,
    SkillEvent,
    SkillExecutionResult,
)


MAX_TOTAL_QUERIES = 6
MAX_EVALUATION_ROUNDS = 3
DEFAULT_AUDIENCE = "SNS・ブログ上の不特定多数"
UNSPECIFIED = "未指定"
LABEL_PATTERN = re.compile(r"^\s*(?:#+\s*)?([^:：\n]{1,30})\s*[:：]\s*(.+?)\s*$")


def run(context: ScriptedSkillContext) -> SkillExecutionResult:
    topic = _extract_topic(context.issue)
    if topic is None:
        return SkillExecutionResult(
            status="needs_user",
            events=(
                SkillEvent(
                    "final_review",
                    "調査テーマを特定できませんでした。チケットに調査テーマを追記してください。",
                ),
            ),
            dry_run=context.dry_run,
        )

    purpose = _extract_labeled_value(context.issue, ("調査目的", "目的", "リサーチ目的")) or UNSPECIFIED
    audience = (
        _extract_labeled_value(context.issue, ("想定読者", "読者", "対象読者"))
        or DEFAULT_AUDIENCE
    )
    events: list[SkillEvent] = []
    collected_results: list[dict[str, Any]] = []
    search_log: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    executed_queries: set[str] = set()

    issue_payload = {
        "issue": _issue_context(context.issue),
        "inferred": {
            "topic": topic,
            "purpose": purpose,
            "audience": audience,
        },
    }
    planned, error = _call(
        context,
        "plan_web_research",
        {"issue_json": json.dumps(issue_payload, ensure_ascii=False)},
    )
    if error is not None:
        return _failed(context, events, "調査計画の作成", error)
    research_plan = _dict(planned.get("plan"))
    if not research_plan:
        return _failed(context, events, "調査計画の作成", "調査計画が返されませんでした。")
    research_plan["topic"] = str(research_plan.get("topic") or topic)
    research_plan["purpose"] = str(research_plan.get("purpose") or purpose)
    research_plan["audience"] = str(research_plan.get("audience") or audience)
    events.append(SkillEvent("progress", _plan_comment(research_plan)))

    pending_queries = _queries(research_plan.get("initial_queries"))
    last_evaluation: dict[str, Any] = {
        "sufficient": False,
        "summary": "根拠評価は未実施です。",
        "covered_points": [],
        "missing_points": ["初期検索前です。"],
        "additional_queries": [],
        "stop_reason": "not_evaluated",
    }

    for round_index in range(1, MAX_EVALUATION_ROUNDS + 1):
        ran_query = False
        while pending_queries and len(executed_queries) < MAX_TOTAL_QUERIES:
            query = pending_queries.pop(0)
            normalized_query = _normalize_query(query["query"])
            if not normalized_query or normalized_query in executed_queries:
                continue
            executed_queries.add(normalized_query)
            ran_query = True
            events.append(
                SkillEvent(
                    "progress",
                    _search_start_comment(
                        len(executed_queries),
                        normalized_query,
                        query["purpose"],
                    ),
                )
            )
            search_result, error = _call(
                context,
                "web_search_pages",
                {
                    "query": normalized_query,
                    "max_results": 5,
                    "page_text_max_chars": 8000,
                    "region": "ja-jp",
                    "safesearch": "moderate",
                    "timelimit": None,
                },
            )
            if error is not None:
                search_log.append(
                    {
                        "query": normalized_query,
                        "purpose": query["purpose"],
                        "ok": False,
                        "error": error,
                    }
                )
                events.append(
                    SkillEvent(
                        "progress",
                        f"Web検索に失敗しました。\n\n- 検索キーワード: {normalized_query}\n- 理由: {error}",
                    )
                )
                continue
            filtered_result = _filter_new_urls(search_result, seen_urls)
            collected_results.append(filtered_result)
            search_log.append(_search_log_row(normalized_query, query["purpose"], filtered_result))
            events.append(SkillEvent("progress", _search_result_comment(filtered_result)))

        if not collected_results:
            if pending_queries and len(executed_queries) >= MAX_TOTAL_QUERIES:
                break
            if not ran_query:
                break

        evaluated, error = _call(
            context,
            "evaluate_research_coverage",
            {
                "research_plan_json": json.dumps(research_plan, ensure_ascii=False),
                "collected_results_json": json.dumps(collected_results, ensure_ascii=False),
                "evaluation_history_json": json.dumps(evaluation_history, ensure_ascii=False),
            },
        )
        if error is not None:
            return _failed(context, events, "根拠十分性の評価", error)
        last_evaluation = _dict(evaluated.get("evaluation"))
        if not last_evaluation:
            return _failed(context, events, "根拠十分性の評価", "評価結果が返されませんでした。")
        evaluation_history.append(last_evaluation)
        events.append(
            SkillEvent(
                "progress",
                _evaluation_comment(round_index, last_evaluation, len(executed_queries)),
            )
        )
        if last_evaluation.get("sufficient") is True:
            break
        if len(executed_queries) >= MAX_TOTAL_QUERIES:
            events.append(
                SkillEvent(
                    "progress",
                    "検索上限に到達したため、残る不足論点は最終レポートの追加調査事項として扱います。",
                )
            )
            break
        pending_queries = [
            query
            for query in _queries(last_evaluation.get("additional_queries"))
            if _normalize_query(query["query"]) not in executed_queries
        ]
        if not pending_queries:
            events.append(
                SkillEvent(
                    "progress",
                    "追加検索クエリがないため、現在の根拠でレポートを作成します。不足論点は明記します。",
                )
            )
            break

    if not collected_results:
        return _failed(context, events, "Web検索", "有効な検索結果を取得できませんでした。")

    events.append(SkillEvent("progress", "最終Markdownレポートを作成します。"))
    composed, error = _call(
        context,
        "compose_research_report",
        {
            "research_plan_json": json.dumps(research_plan, ensure_ascii=False),
            "collected_results_json": json.dumps(collected_results, ensure_ascii=False),
            "coverage_json": json.dumps(last_evaluation, ensure_ascii=False),
            "search_log_json": json.dumps(search_log, ensure_ascii=False),
        },
    )
    if error is not None:
        return _failed(context, events, "リサーチレポートの作成", error)
    report = _string(composed.get("report"))
    if report is None:
        return _failed(context, events, "リサーチレポートの作成", "レポート本文が返されませんでした。")

    status = "dry_run" if context.dry_run else "processed"
    return SkillExecutionResult(
        status=status,
        events=(*events, SkillEvent("final_review", report)),
        artifacts=(
            {
                "type": "web_research_report",
                "research_plan": research_plan,
                "search_log": search_log,
                "coverage": last_evaluation,
            },
        ),
        dry_run=context.dry_run,
    )


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


def _failed(
    context: ScriptedSkillContext,
    events: list[SkillEvent],
    step: str,
    error: str,
) -> SkillExecutionResult:
    return SkillExecutionResult(
        status="failed",
        events=(
            *events,
            SkillEvent(
                "final_return",
                f"{step}に失敗したため、後続処理を停止します。\n理由: {error}",
            ),
        ),
        dry_run=context.dry_run,
    )


def _extract_topic(issue: dict[str, Any]) -> str | None:
    explicit = _extract_labeled_value(issue, ("調査テーマ", "テーマ", "リサーチテーマ"))
    if explicit:
        return explicit
    subject = _string(issue.get("subject"))
    if subject and not _is_generic_topic(subject):
        return subject
    description = _string(issue.get("description"))
    if not description:
        return None
    for raw_line in description.splitlines():
        line = _clean_markdown_line(raw_line)
        if not line or _looks_like_instruction_only(line):
            continue
        match = LABEL_PATTERN.match(line)
        if match:
            label = match.group(1).strip()
            if label in {"調査目的", "目的", "想定読者", "対象期間", "対象地域", "対象領域", "除外範囲"}:
                continue
            return match.group(2).strip()
        return line
    return None


def _extract_labeled_value(issue: dict[str, Any], labels: tuple[str, ...]) -> str | None:
    description = _string(issue.get("description")) or ""
    for raw_line in description.splitlines():
        match = LABEL_PATTERN.match(_clean_markdown_line(raw_line))
        if not match:
            continue
        label = match.group(1).strip()
        if label in labels:
            value = match.group(2).strip()
            if value:
                return value
    return None


def _clean_markdown_line(value: str) -> str:
    line = value.strip()
    line = re.sub(r"^[-*]\s+", "", line)
    return line.strip()


def _looks_like_instruction_only(value: str) -> bool:
    return value in {
        "リサーチレポートを作成してください",
        "調査してください",
        "レポートを作成してください",
    }


def _is_generic_topic(value: str) -> bool:
    stripped = value.strip()
    return stripped in {"調査", "リサーチ", "レポート作成", "リサーチレポート作成"}


def _issue_context(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": issue.get("id"),
        "subject": issue.get("subject"),
        "description": issue.get("description"),
        "author": issue.get("author"),
        "status": issue.get("status"),
        "priority": issue.get("priority"),
        "project": issue.get("project"),
        "tracker": issue.get("tracker"),
    }


def _queries(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        query = _normalize_query(item.get("query"))
        purpose = _string(item.get("purpose")) or "関連情報を収集する"
        if not query or query in seen:
            continue
        seen.add(query)
        queries.append({"query": query, "purpose": purpose})
    return queries


def _normalize_query(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    query = re.sub(r"\s+", " ", value).strip()
    return query or None


def _filter_new_urls(result: dict[str, Any], seen_urls: set[str]) -> dict[str, Any]:
    pages = result.get("pages")
    if not isinstance(pages, list):
        pages = []
    kept_pages: list[dict[str, Any]] = []
    kept_ranks: set[Any] = set()
    for page in pages:
        if not isinstance(page, dict):
            continue
        url = _canonical_url(page.get("final_url") or page.get("url"))
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        kept_pages.append(page)
        kept_ranks.add(page.get("rank"))

    search_results = result.get("search_results")
    if isinstance(search_results, list):
        kept_search_results = [
            item
            for item in search_results
            if isinstance(item, dict) and item.get("rank") in kept_ranks
        ]
    else:
        kept_search_results = []

    filtered = dict(result)
    filtered["search_results"] = kept_search_results
    filtered["pages"] = kept_pages
    filtered["deduped_count"] = len(pages) - len(kept_pages)
    return filtered


def _canonical_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value.rstrip("/")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def _plan_comment(plan: dict[str, Any]) -> str:
    lines = [
        "Webリサーチの調査計画を作成しました。",
        "",
        f"- 調査テーマ: {plan.get('topic') or UNSPECIFIED}",
        f"- 調査目的: {plan.get('purpose') or UNSPECIFIED}",
        f"- 想定読者: {plan.get('audience') or DEFAULT_AUDIENCE}",
        f"- 対象期間: {plan.get('target_period') or UNSPECIFIED}",
        f"- 対象地域: {plan.get('target_region') or UNSPECIFIED}",
        f"- 対象領域: {plan.get('target_domain') or UNSPECIFIED}",
        "",
        "リサーチクエスチョン:",
        f"- メイン: {plan.get('main_question') or UNSPECIFIED}",
    ]
    for question in plan.get("sub_questions") or []:
        lines.append(f"- サブ: {question}")
    lines.extend(["", "初期検索キーワード:"])
    for query in _queries(plan.get("initial_queries")):
        lines.append(f"- {query['query']}（目的: {query['purpose']}）")
    return "\n".join(lines)


def _search_start_comment(index: int, query: str, purpose: str) -> str:
    return (
        f"Web検索を実行します（{index}/{MAX_TOTAL_QUERIES}）。\n\n"
        f"- 検索キーワード: {query}\n"
        f"- 検索目的: {purpose}"
    )


def _search_result_comment(result: dict[str, Any]) -> str:
    query = result.get("query") or "(未記録)"
    search_results = result.get("search_results")
    pages = result.get("pages")
    search_count = len(search_results) if isinstance(search_results, list) else 0
    page_rows = pages if isinstance(pages, list) else []
    success_count = sum(1 for page in page_rows if isinstance(page, dict) and page.get("fetch_ok") is True)
    failure_count = sum(1 for page in page_rows if isinstance(page, dict) and page.get("fetch_ok") is not True)
    lines = [
        "Web検索結果を取得しました。",
        "",
        f"- 検索キーワード: {query}",
        f"- 検索結果: {search_count}件",
        f"- 本文取得成功: {success_count}件",
        f"- 本文取得失敗: {failure_count}件",
    ]
    deduped = result.get("deduped_count")
    if isinstance(deduped, int) and deduped:
        lines.append(f"- 取得済みURLのため除外: {deduped}件")
    lines.append("- 主要URL:")
    urls = [
        page.get("final_url") or page.get("url")
        for page in page_rows
        if isinstance(page, dict) and (page.get("final_url") or page.get("url"))
    ]
    lines.extend(f"  - {url}" for url in urls[:3])
    if not urls:
        lines.append("  - なし")
    return "\n".join(lines)


def _evaluation_comment(round_index: int, evaluation: dict[str, Any], query_count: int) -> str:
    sufficient = "十分" if evaluation.get("sufficient") is True else "不足あり"
    lines = [
        f"根拠十分性を評価しました（評価ラウンド {round_index}/{MAX_EVALUATION_ROUNDS}）。",
        "",
        f"- 判定: {sufficient}",
        f"- 実行済み検索数: {query_count}/{MAX_TOTAL_QUERIES}",
        f"- 評価要約: {evaluation.get('summary') or UNSPECIFIED}",
    ]
    covered = evaluation.get("covered_points") or []
    missing = evaluation.get("missing_points") or []
    additional = _queries(evaluation.get("additional_queries"))
    lines.append("- カバー済み論点:")
    lines.extend(f"  - {item}" for item in covered) if covered else lines.append("  - なし")
    lines.append("- 不足論点:")
    lines.extend(f"  - {item}" for item in missing) if missing else lines.append("  - なし")
    lines.append("- 追加検索キーワード:")
    lines.extend(
        f"  - {query['query']}（目的: {query['purpose']}）"
        for query in additional
    ) if additional else lines.append("  - なし")
    return "\n".join(lines)


def _search_log_row(query: str, purpose: str, result: dict[str, Any]) -> dict[str, Any]:
    pages = result.get("pages")
    page_rows = pages if isinstance(pages, list) else []
    return {
        "query": query,
        "purpose": purpose,
        "ok": True,
        "search_result_count": len(result.get("search_results") or []),
        "fetch_success_count": sum(
            1 for page in page_rows if isinstance(page, dict) and page.get("fetch_ok") is True
        ),
        "fetch_failure_count": sum(
            1 for page in page_rows if isinstance(page, dict) and page.get("fetch_ok") is not True
        ),
        "urls": [
            page.get("final_url") or page.get("url")
            for page in page_rows
            if isinstance(page, dict) and (page.get("final_url") or page.get("url"))
        ],
    }


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
