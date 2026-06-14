from __future__ import annotations

from typing import Any

from taskboard_agent.tool_loader import ToolRuntimeContext
from taskboard_agent.tools import ToolSpec


TOOL_SPEC = ToolSpec(
    name="fetch_web_page",
    description="指定URLのWebページから最終URL、タイトル、本文を抽出する。",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    },
    risk="read",
)


def create_handler(context: ToolRuntimeContext) -> Any:
    page_fetcher = context.require_service("page_fetcher")

    def handle(*, url: str) -> dict[str, Any]:
        try:
            page = page_fetcher.fetch(url)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": url}
        return {"ok": True, "url": page.url, "title": page.title, "text": page.text}

    return handle
