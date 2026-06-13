from __future__ import annotations

import argparse
import sys

from taskboard_agent.config import ConfigError, load_config
from taskboard_agent.linkace import LinkAceClient, LinkAceError
from taskboard_agent.llm import (
    CommentGenerationError,
    OpenAIBriefingSummarizer,
    OpenAIRequestClassifier,
)
from taskboard_agent.page import PageFetchError, WebPageExtractor
from taskboard_agent.redmine import RedmineClient, RedmineError
from taskboard_agent.workflow import WorkflowError, run_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="taskboard-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_once_parser = subparsers.add_parser(
        "run-once",
        help="Process one open Redmine issue assigned to the AI user.",
    )
    run_once_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fetch the issue, page, and briefing, but do not update Redmine or LinkAce."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run-once":
        parser.error(f"unknown command: {args.command}")

    try:
        config = load_config()
        redmine = RedmineClient(config.redmine_url, config.redmine_api_key)
        request_classifier = OpenAIRequestClassifier(
            api_key=config.openai_api_key,
            model=config.openai_model,
        )
        page_fetcher = WebPageExtractor()
        briefing_summarizer = OpenAIBriefingSummarizer(
            api_key=config.openai_api_key,
            model=config.openai_model,
        )
        bookmark_client = LinkAceClient(config.linkace_url, config.linkace_api_key)
        result = run_once(
            config=config,
            redmine=redmine,
            request_classifier=request_classifier,
            page_fetcher=page_fetcher,
            briefing_summarizer=briefing_summarizer,
            bookmark_client=bookmark_client,
            dry_run=args.dry_run,
        )
    except (
        ConfigError,
        CommentGenerationError,
        LinkAceError,
        PageFetchError,
        RedmineError,
        WorkflowError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.status == "no_issue":
        print("No open Redmine issues are assigned to the AI user.")
        return 0

    if result.dry_run:
        print(
            f"Dry run complete for issue #{result.issue_id}; Redmine and LinkAce were not updated."
        )
        if result.target_url:
            print()
            print(f"Target URL: {result.target_url}")
        if result.page_title:
            print(f"Page title: {result.page_title}")
        if result.briefing:
            print()
            print("Generated briefing:")
            print(result.briefing)
        if result.bookmark_payload:
            print()
            print("LinkAce payload:")
            print(result.bookmark_payload)
        if result.comments:
            print()
            print("Comments that would be posted:")
            for comment in result.comments:
                print("---")
                print(comment)
        return 0

    print(
        "Processed issue "
        f"#{result.issue_id}; generated briefing, registered bookmark, and "
        f"reassigned to author #{result.reassigned_to_id}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
