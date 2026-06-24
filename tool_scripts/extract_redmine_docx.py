from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from taskboard_agent.docx import extract_docx_text
from taskboard_agent.tool_loader import ToolRuntimeContext


def create_tool(context: ToolRuntimeContext) -> BaseTool:
    redmine_client = context.require_service("redmine_client")

    @tool(
        parse_docstring=True,
        extras={"risk": "read", "planner_visible": False},
    )
    def extract_redmine_docx(filename: str, content_url: str) -> dict[str, Any]:
        """RedmineのDOCX添付を取得し、段落と入れ子表を文書順のテキストへ復元する。

        Args:
            filename: 添付ファイル名。
            content_url: Redmine添付ファイル本文の取得URL。
        """
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

    return extract_redmine_docx
