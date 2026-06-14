from __future__ import annotations

from typing import Any

from taskboard_agent.tool_loader import ToolRuntimeContext
from taskboard_agent.tools import ToolSpec


BRIEFING_PROMPT = (
    "情報源から主要なテーマとアイデアを統合した包括的なブリーフィングドキュメントを作成してください。"
    "まずは、最も重要なポイントを簡潔にまとめたエグゼクティブサマリーから始めましょう。"
    "本文では、情報源に含まれる主要なテーマ、証拠、そして結論を​​詳細かつ徹底的に検証する必要があります。"
    "分析は、明瞭性を確保するために、見出しと箇条書きを用いて論理的に構成する必要があります。"
    "トーンは客観的かつ鋭いものでなければなりません。"
)
MAX_INPUT_CHARS = 60000

TOOL_SPEC = ToolSpec(
    name="summarize_briefing",
    description="Webページ本文をブリーフィング要約する。",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "title": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["url", "title", "text"],
        "additionalProperties": False,
    },
    risk="read",
)


def create_handler(context: ToolRuntimeContext) -> Any:
    llm = context.require_service("llm")

    def handle(*, url: str, title: str, text: str) -> dict[str, Any]:
        try:
            response = llm.complete(
                [
                    {"role": "system", "content": BRIEFING_PROMPT},
                    {
                        "role": "user",
                        "content": _build_prompt(
                            url=url,
                            title=title,
                            text=text[:MAX_INPUT_CHARS],
                        ),
                    },
                ]
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": url, "title": title}
        briefing = response.content.strip()
        if not briefing:
            return {
                "ok": False,
                "error": "generated briefing was empty",
                "url": url,
                "title": title,
            }
        return {"ok": True, "url": url, "title": title, "briefing": briefing}

    return handle


def _build_prompt(*, url: str, title: str, text: str) -> str:
    return (
        f"{BRIEFING_PROMPT}\n\n"
        f"URL: {url}\n"
        f"タイトル: {title}\n\n"
        "抽出本文:\n"
        "<<<ARTICLE_TEXT\n"
        f"{text}\n"
        "ARTICLE_TEXT"
    )
