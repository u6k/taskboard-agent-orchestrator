from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


class CommentGenerationError(RuntimeError):
    """Raised when an updated issue description cannot be generated."""


@dataclass(frozen=True)
class RequestClassification:
    can_handle: bool
    url: str | None
    reason: str


BRIEFING_PROMPT = (
    "情報源から主要なテーマとアイデアを統合した包括的なブリーフィングドキュメントを作成してください。"
    "まずは、最も重要なポイントを簡潔にまとめたエグゼクティブサマリーから始めましょう。"
    "本文では、情報源に含まれる主要なテーマ、証拠、そして結論を​​詳細かつ徹底的に検証する必要があります。"
    "分析は、明瞭性を確保するために、見出しと箇条書きを用いて論理的に構成する必要があります。"
    "トーンは客観的かつ鋭いものでなければなりません。"
)


class OpenAIRequestClassifier:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def classify(self, issue: dict[str, Any]) -> RequestClassification:
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=(
                    "あなたはRedmineチケットを読む業務AIです。"
                    "チケットが、指定URLのWebページ本文をブリーフィング要約し、"
                    "ブックマーク登録する依頼かどうかだけを判定してください。"
                    "出力はJSONのみです。"
                ),
                input=build_request_classification_prompt(issue),
            )
        except Exception as exc:  # pragma: no cover - SDK exceptions vary by version.
            raise CommentGenerationError(f"failed to classify request: {exc}") from exc

        return parse_request_classification(_extract_output_text(response))


class OpenAIBriefingSummarizer:
    def __init__(self, api_key: str, model: str, *, max_input_chars: int = 60000) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_input_chars = max_input_chars

    def summarize(self, *, url: str, title: str, text: str) -> str:
        prompt = build_briefing_prompt(
            url=url,
            title=title,
            text=text[: self._max_input_chars],
        )
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=BRIEFING_PROMPT,
                input=prompt,
            )
        except Exception as exc:  # pragma: no cover - SDK exceptions vary by version.
            raise CommentGenerationError(f"failed to generate briefing: {exc}") from exc

        briefing = _extract_output_text(response)
        if briefing.strip() == "":
            raise CommentGenerationError("generated briefing was empty")
        return briefing.strip()


class OpenAIDescriptionGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, issue: dict[str, Any]) -> str:
        prompt = build_issue_prompt(issue)
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=(
                    "あなたはRedmineチケットを読む業務AIです。"
                    "作業は実行せず、チケットのDescriptionを日本語で整理してください。"
                    "目的、作業内容、完了条件、課題を簡潔に整理し、推測は課題に含めてください。"
                    "ユーザーが記載した元文章は省略・改変せず、最後の見出しに残してください。"
                ),
                input=prompt,
            )
        except Exception as exc:  # pragma: no cover - SDK exceptions vary by version.
            raise CommentGenerationError(f"failed to generate comment: {exc}") from exc

        comment = _extract_output_text(response)
        if comment.strip() == "":
            raise CommentGenerationError("generated comment was empty")
        return comment.strip()


def build_request_classification_prompt(issue: dict[str, Any]) -> str:
    return (
        "次のRedmineチケットが処理対象か判定してください。\n"
        "処理対象は「指定URLのWebページ本文をブリーフィング要約し、"
        "ブックマーク登録する依頼」だけです。\n"
        "URLが複数ある場合は、依頼対象として最も明確な1件をurlに入れてください。\n"
        "判定不能、URLなし、別種の依頼なら can_handle=false にしてください。\n"
        "必ず次のJSONだけを出力してください。\n"
        '{"can_handle": true|false, "url": "https://example.com または null", '
        '"reason": "判定理由"}\n\n'
        f"件名:\n{issue.get('subject') or '(未記載)'}\n\n"
        "詳細:\n"
        "<<<DESCRIPTION\n"
        f"{issue.get('description') or '(未記載)'}\n"
        "DESCRIPTION"
    )


def parse_request_classification(output: str) -> RequestClassification:
    try:
        data = json.loads(_strip_json_fence(output))
    except json.JSONDecodeError as exc:
        raise CommentGenerationError("request classification was not valid JSON") from exc
    if not isinstance(data, dict):
        raise CommentGenerationError("request classification JSON was not an object")

    can_handle = data.get("can_handle")
    if not isinstance(can_handle, bool):
        raise CommentGenerationError("request classification missing boolean can_handle")

    url_value = data.get("url")
    url = url_value.strip() if isinstance(url_value, str) else None
    if url == "":
        url = None
    if can_handle and url is None:
        raise CommentGenerationError("request classification can_handle=true without url")

    reason_value = data.get("reason")
    reason = reason_value.strip() if isinstance(reason_value, str) else ""
    return RequestClassification(can_handle=can_handle, url=url, reason=reason)


def build_briefing_prompt(*, url: str, title: str, text: str) -> str:
    return (
        f"{BRIEFING_PROMPT}\n\n"
        f"URL: {url}\n"
        f"タイトル: {title}\n\n"
        "抽出本文:\n"
        "<<<ARTICLE_TEXT\n"
        f"{text}\n"
        "ARTICLE_TEXT"
    )


def build_issue_prompt(issue: dict[str, Any]) -> str:
    original_description = issue.get("description") or "(未記載)"
    fields = {
        "ID": issue.get("id"),
        "題名": issue.get("subject"),
        "作成者": _named_value(issue.get("author")),
        "担当者": _named_value(issue.get("assigned_to")),
        "ステータス": _named_value(issue.get("status")),
        "優先度": _named_value(issue.get("priority")),
        "プロジェクト": _named_value(issue.get("project")),
        "トラッカー": _named_value(issue.get("tracker")),
        "開始日": issue.get("start_date"),
        "期日": issue.get("due_date"),
    }
    field_text = "\n".join(f"- {key}: {value or '(未設定)'}" for key, value in fields.items())

    return (
        "次のRedmineチケットのDescriptionを更新する本文だけを出力してください。\n"
        "余計な前置き、コードブロック、説明文は出力しないでください。\n"
        "フォーマットは必ず次の形にしてください。\n\n"
        "# 目的\n"
        "{目的}\n\n"
        "# 実施すべき内容\n"
        "- {作業内容1}\n"
        "- {作業内容2}\n\n"
        "# 完了条件\n"
        "- {完了条件1}\n"
        "- {完了条件2}\n\n"
        "# 課題\n"
        "- {課題点、不明点、確認したい点}\n\n"
        "# ユーザーが記載した元文章\n"
        "{もともとDescriptionにユーザーが記載していた文章}\n\n"
        "元の説明文が空の場合は、最後の見出しの本文を「(未記載)」にしてください。\n\n"
        f"チケット情報:\n{field_text}\n\n"
        "ユーザーが記載した元Description:\n"
        "<<<ORIGINAL_DESCRIPTION\n"
        f"{original_description}\n"
        "ORIGINAL_DESCRIPTION"
    )


def _named_value(value: Any) -> str | None:
    if isinstance(value, dict):
        name = value.get("name")
        identifier = value.get("id")
        if name and identifier:
            return f"{name} (ID: {identifier})"
        if name:
            return str(name)
        if identifier:
            return f"ID: {identifier}"
    if value is None:
        return None
    return str(value)


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _strip_json_fence(output: str) -> str:
    stripped = output.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped
