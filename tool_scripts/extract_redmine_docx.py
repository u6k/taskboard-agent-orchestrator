from __future__ import annotations

from typing import Any

from taskboard_agent.docx import extract_docx_text
from taskboard_agent.tool_loader import ToolRuntimeContext
from taskboard_agent.tools import ToolSpec


TOOL_SPEC = ToolSpec(
    name="extract_redmine_docx",
    description="RedmineのDOCX添付を取得し、段落と入れ子表を文書順のテキストへ復元する。",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string"},
            "content_url": {"type": "string"},
        },
        "required": ["filename", "content_url"],
        "additionalProperties": False,
    },
    risk="read",
    planner_visible=False,
)


def create_handler(context: ToolRuntimeContext) -> Any:
    redmine_client = context.require_service("redmine_client")

    def handle(*, filename: str, content_url: str) -> dict[str, Any]:
        if not filename.lower().endswith(".docx"):
            return {"ok": False, "filename": filename, "error": "DOCXファイルではありません。"}
        try:
            content = redmine_client.download_attachment(content_url)
            text = extract_docx_text(content)
        except Exception as exc:
            return {"ok": False, "filename": filename, "error": str(exc)}
        return {
            "ok": True,
            "filename": filename,
            "text": text,
            "character_count": len(text),
        }

    return handle
