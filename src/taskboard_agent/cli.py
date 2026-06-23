from __future__ import annotations

import argparse
from contextlib import nullcontext
import logging
import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from taskboard_agent.agent import FunctionCallingAgent
from taskboard_agent.config import ConfigError, load_config
from taskboard_agent.linkace import LinkAceClient, LinkAceError
from taskboard_agent.logging_config import configure_logging, log_trace
from taskboard_agent.llm import LiteLLMClient
from taskboard_agent.page import PageFetchError, WebPageExtractor
from taskboard_agent.redmine import RedmineClient, RedmineError
from taskboard_agent.skills import SkillRegistry, SkillRegistryError
from taskboard_agent.task_executor import (
    GenericTaskRunner,
    LiteLLMTaskPlanner,
    TaskOrchestrator,
    TaskPlanningError,
)
from taskboard_agent.ticket_graph import TicketConversationGraph
from taskboard_agent.tool_loader import ToolRuntimeContext, ToolScriptCatalog
from taskboard_agent.web_search import DuckDuckGoSearchClient, WebSearchError
from taskboard_agent.workflow import WorkflowError, run_once


logger = logging.getLogger(__name__)


def _positive_issue_id(value: str) -> int:
    try:
        issue_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("issue ID must be an integer") from exc
    if issue_id <= 0:
        raise argparse.ArgumentTypeError("issue ID must be a positive integer")
    return issue_id


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
            "Understand and execute the issue without updating Redmine or external services."
        ),
    )
    run_once_parser.add_argument(
        "--issue-id",
        type=_positive_issue_id,
        help=(
            "Process this Redmine issue directly instead of searching for an open "
            "issue assigned to the AI user."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging("logging.conf")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run-once":
        parser.error(f"unknown command: {args.command}")

    with log_trace("run-once"):
        logger.info("CLIを開始します command=%s dry_run=%s", args.command, args.dry_run)
    try:
        config = load_config()
        redmine = RedmineClient(config.redmine_url, config.redmine_api_key)
        llm = LiteLLMClient(model=config.llm_model)
        page_fetcher = WebPageExtractor()
        search_client = DuckDuckGoSearchClient()
        bookmark_client = LinkAceClient(config.linkace_url, config.linkace_api_key)
        skill_registry = SkillRegistry(Path("skills"))
        skill_agent = FunctionCallingAgent(llm=llm)
        tool_catalog = ToolScriptCatalog(
            Path("tool_scripts"),
            ToolRuntimeContext(
                services={
                    "llm": llm,
                    "page_fetcher": page_fetcher,
                    "search_client": search_client,
                    "bookmark_client": bookmark_client,
                    "redmine_client": redmine,
                },
                settings={
                    "linkace_summarized_list_id": config.linkace_summarized_list_id,
                },
                dry_run=args.dry_run,
            ),
        )
        task_orchestrator = TaskOrchestrator(
            planner=LiteLLMTaskPlanner(llm),
            skill_registry=skill_registry,
            tool_catalog=tool_catalog,
            skill_agent=skill_agent,
            generic_runner=GenericTaskRunner(llm),
        )
        if args.dry_run:
            checkpointer_context = nullcontext(InMemorySaver())
        else:
            config.langgraph_checkpoint_db_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            checkpointer_context = SqliteSaver.from_conn_string(
                str(config.langgraph_checkpoint_db_path)
            )
        with checkpointer_context as checkpointer:
            task_executor = TicketConversationGraph(
                task_orchestrator=task_orchestrator,
                llm=llm,
                checkpointer=checkpointer,
                ai_user_id=config.redmine_ai_user_id,
            )
            result = run_once(
                config=config,
                redmine=redmine,
                task_executor=task_executor,
                dry_run=args.dry_run,
                issue_id=args.issue_id,
            )
    except (
        ConfigError,
        LinkAceError,
        PageFetchError,
        RedmineError,
        SkillRegistryError,
        TaskPlanningError,
        WebSearchError,
        WorkflowError,
    ) as exc:
        with log_trace("run-once"):
            logger.warning("CLI実行中に例外が発生しました", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.status == "no_issue":
        with log_trace("run-once"):
            logger.info("CLIを終了します status=no_issue")
        print("No open Redmine issues are assigned to the AI user.")
        return 0

    if result.dry_run:
        with log_trace(f"issue#{result.issue_id}" if result.issue_id else "run-once"):
            logger.info("CLIを終了します status=%s dry_run=True", result.status)
        print(
            f"Dry run complete for issue #{result.issue_id}; external services were not updated."
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
            print("Tool payload:")
            print(result.bookmark_payload)
        if result.comments:
            print()
            print("Comments that would be posted:")
            for comment in result.comments:
                print("---")
                print(comment)
        return 0

    with log_trace(f"issue#{result.issue_id}" if result.issue_id else "run-once"):
        logger.info(
            "CLIを終了します status=%s reassigned_to_id=%s",
            result.status,
            result.reassigned_to_id,
        )
    print(
        "Processed issue "
        f"#{result.issue_id}; status={result.status}; "
        f"reassigned to author #{result.reassigned_to_id}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
