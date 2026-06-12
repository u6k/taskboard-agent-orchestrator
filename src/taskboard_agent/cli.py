from __future__ import annotations

import argparse
import sys

from taskboard_agent.config import ConfigError, load_config
from taskboard_agent.llm import CommentGenerationError, OpenAIDescriptionGenerator
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
            "Fetch the issue and generate the updated description/comment, "
            "but do not update Redmine."
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
        description_generator = OpenAIDescriptionGenerator(
            api_key=config.openai_api_key,
            model=config.openai_model,
        )
        result = run_once(
            config=config,
            redmine=redmine,
            description_generator=description_generator,
            dry_run=args.dry_run,
        )
    except (ConfigError, CommentGenerationError, RedmineError, WorkflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.status == "no_issue":
        print("No open Redmine issues are assigned to the AI user.")
        return 0

    if result.dry_run:
        print(
            f"Dry run complete for issue #{result.issue_id}; Redmine was not updated."
        )
        if result.description:
            print()
            print("Generated description:")
            print(result.description)
        if result.comment:
            print()
            print("Generated comment:")
            print(result.comment)
        return 0

    print(
        "Processed issue "
        f"#{result.issue_id}; updated description, added action comment, and "
        f"reassigned to author #{result.reassigned_to_id}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
