from __future__ import annotations

from typing import Any

from openai import OpenAI


class CommentGenerationError(RuntimeError):
    """Raised when an updated issue description cannot be generated."""


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
