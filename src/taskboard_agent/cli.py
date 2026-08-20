from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_litellm import ChatLiteLLM

from taskboard_agent.agent import LangChainAgentRunner
from taskboard_agent.artifacts import FileArtifactStore, InMemoryArtifactStore
from taskboard_agent.config import AppConfig, ConfigError, load_config
from taskboard_agent.context_engine import ContextEngine
from taskboard_agent.daemon import AgentExecutionContext, DaemonResult, run_daemon
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
from taskboard_agent.workflow import RunResult, WorkflowError, run_once


logger = logging.getLogger(__name__)


class TaskboardArgumentParser(argparse.ArgumentParser):
    def parse_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        if (
            parsed.command == "run-daemon"
            and parsed.dry_run
            and parsed.max_iterations is None
        ):
            self.error("run-daemon --dry-run requires --max-iterations")
        return parsed


@dataclass(frozen=True)
class Runtime:
    config: AppConfig
    agents: tuple[AgentExecutionContext, ...]


def _positive_issue_id(value: str) -> int:
    try:
        issue_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("issue ID must be an integer") from exc
    if issue_id <= 0:
        raise argparse.ArgumentTypeError("issue ID must be a positive integer")
    return issue_id


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = TaskboardArgumentParser(prog="taskboard-agent")
    parser.add_argument(
        "--config",
        default="agents.toml",
        help="Agent profile TOML path. Defaults to agents.toml.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_once_parser = subparsers.add_parser(
        "run-once",
        help="Process one open Redmine issue for the selected agent profile.",
    )
    run_once_parser.add_argument(
        "--agent",
        required=True,
        help="Agent profile ID to use for this run.",
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
            "issue assigned to the selected agent profile."
        ),
    )

    run_daemon_parser = subparsers.add_parser(
        "run-daemon",
        help="Continuously poll and process issues for all enabled agent profiles.",
    )
    run_daemon_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Understand and execute issues without updating Redmine or external "
            "services. Requires --max-iterations."
        ),
    )
    run_daemon_parser.add_argument(
        "--interval-seconds",
        type=_positive_int,
        default=60,
        help="Polling interval used only when no assigned issue is found. Defaults to 60.",
    )
    run_daemon_parser.add_argument(
        "--max-iterations",
        type=_positive_int,
        help="Stop after this many polling loop iterations. Intended for tests and checks.",
    )

    return parser


@contextmanager
def build_runtime(*, dry_run: bool, config_path: str | Path = "agents.toml") -> Iterator[Runtime]:
    config = load_config(agents_file=config_path)
    page_fetcher = WebPageExtractor()
    search_client = DuckDuckGoSearchClient()
    bookmark_client = LinkAceClient(config.linkace_url, config.linkace_api_key)
    skill_registry = SkillRegistry(Path("skills"))
    if dry_run:
        checkpointer_context = nullcontext(InMemorySaver())
        artifact_store = InMemoryArtifactStore()
    else:
        config.langgraph_checkpoint_db_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        checkpointer_context = SqliteSaver.from_conn_string(
            str(config.langgraph_checkpoint_db_path)
        )
        artifact_store = FileArtifactStore(
            config.langgraph_checkpoint_db_path.parent / "artifacts"
        )
    with checkpointer_context as checkpointer:
        ai_user_ids = {agent.redmine_user_id for agent in config.agents}
        agent_contexts: list[AgentExecutionContext] = []
        for profile in config.agents:
            logger.info(
                "エージェントruntimeを構築します agent_id=%s "
                "redmine_user_id=%s model=%s api_base=%s timeout_seconds=%s context_window_tokens=%s",
                profile.id,
                profile.redmine_user_id,
                profile.llm_model,
                profile.llm_api_base,
                profile.llm_timeout_seconds,
                profile.context_window_tokens,
            )
            redmine = RedmineClient(config.redmine_url, profile.redmine_api_key)
            llm = LiteLLMClient(
                model=profile.llm_model,
                api_base=profile.llm_api_base,
                api_key=profile.llm_api_key,
                timeout_seconds=profile.llm_timeout_seconds,
                system_prompt=profile.system_prompt,
            )
            chat_model = ChatLiteLLM(
                model=profile.llm_model,
                api_base=profile.llm_api_base,
                api_key=profile.llm_api_key,
                request_timeout=profile.llm_timeout_seconds,
            )
            skill_agent = LangChainAgentRunner(
                model=chat_model,
                system_prompt=profile.system_prompt,
            )
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
                    dry_run=dry_run,
                ),
            )
            task_orchestrator = TaskOrchestrator(
                planner=LiteLLMTaskPlanner(llm),
                skill_registry=skill_registry,
                tool_catalog=tool_catalog,
                skill_agent=skill_agent,
                generic_runner=GenericTaskRunner(llm),
            )
            task_executor = TicketConversationGraph(
                task_orchestrator=task_orchestrator,
                llm=llm,
                checkpointer=checkpointer,
                ai_user_ids=ai_user_ids,
                context_engine=ContextEngine(
                    llm=llm,
                    artifact_store=artifact_store,
                    context_window_tokens=profile.context_window_tokens,
                ),
                artifact_store=artifact_store,
            )
            agent_contexts.append(
                AgentExecutionContext(
                    profile=profile,
                    redmine=redmine,
                    task_executor=task_executor,
                )
            )
        yield Runtime(
            config=config,
            agents=tuple(agent_contexts),
        )


def main(argv: list[str] | None = None) -> int:
    configure_logging("logging.conf")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run-once":
            with log_trace("run-once"):
                logger.info(
                    "CLIを開始します command=%s dry_run=%s",
                    args.command,
                    args.dry_run,
                )
            with build_runtime(
                dry_run=args.dry_run, config_path=args.config
            ) as runtime:
                agent_context = _agent_context(runtime, args.agent)
                result = run_once(
                    config=runtime.config,
                    agent=agent_context.profile,
                    redmine=agent_context.redmine,
                    task_executor=agent_context.task_executor,
                    dry_run=args.dry_run,
                    issue_id=args.issue_id,
                )
            return _print_run_once_result(result)

        if args.command == "run-daemon":
            with log_trace("run-daemon"):
                logger.info(
                    "CLIを開始します command=%s dry_run=%s interval_seconds=%s max_iterations=%s",
                    args.command,
                    args.dry_run,
                    args.interval_seconds,
                    args.max_iterations,
                )
            with build_runtime(
                dry_run=args.dry_run, config_path=args.config
            ) as runtime:
                daemon_result = run_daemon(
                    config=runtime.config,
                    agents=runtime.agents,
                    dry_run=args.dry_run,
                    interval_seconds=args.interval_seconds,
                    max_iterations=args.max_iterations,
                )
            _print_daemon_result(daemon_result)
            return 0

        parser.error(f"unknown command: {args.command}")
        return 2
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
        with log_trace(args.command):
            logger.warning("CLI実行中に例外が発生しました", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _print_run_once_result(result: RunResult) -> int:
    if result.status == "no_issue":
        with log_trace("run-once"):
            logger.info("CLIを終了します status=no_issue")
        print(f"No open Redmine issues are assigned to agent {result.agent_id}.")
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
        f"#{result.issue_id}; agent={result.agent_id}; status={result.status}; "
        f"reassigned to author #{result.reassigned_to_id}."
    )
    return 0


def _print_daemon_result(result: DaemonResult) -> None:
    print(
        "Daemon stopped; "
        f"iterations={result.iterations}; "
        f"processed={result.processed}; "
        f"no_issue={result.no_issue}; "
        f"stopped_by_signal={result.stopped_by_signal}."
    )


def _agent_context(runtime: Runtime, agent_id: str) -> AgentExecutionContext:
    for context in runtime.agents:
        if context.profile.id == agent_id:
            return context
    available = ", ".join(context.profile.id for context in runtime.agents)
    raise ConfigError(
        f"unknown agent profile: {agent_id}; available profiles: {available}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
