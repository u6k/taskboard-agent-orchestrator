from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from types import FrameType
from typing import Callable, Protocol

from taskboard_agent.config import AgentProfileConfig, AppConfig
from taskboard_agent.workflow import (
    RedminePort,
    RunResult,
    TaskExecutorPort,
    run_once,
)


logger = logging.getLogger(__name__)


class RunOnceFunc(Protocol):
    def __call__(
        self,
        *,
        config: AppConfig,
        agent: AgentProfileConfig,
        redmine: RedminePort,
        task_executor: TaskExecutorPort,
        dry_run: bool = False,
        issue_id: int | None = None,
    ) -> RunResult:
        ...


@dataclass(frozen=True)
class DaemonResult:
    iterations: int
    processed: int
    no_issue: int
    stopped_by_signal: bool = False


@dataclass(frozen=True)
class AgentExecutionContext:
    profile: AgentProfileConfig
    redmine: RedminePort
    task_executor: TaskExecutorPort


def run_daemon(
    *,
    config: AppConfig,
    agents: tuple[AgentExecutionContext, ...],
    dry_run: bool = False,
    interval_seconds: int = 60,
    max_iterations: int | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    run_once_func: RunOnceFunc = run_once,
    install_signal_handlers: bool = True,
) -> DaemonResult:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be a positive integer")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not agents:
        raise ValueError("agents must not be empty")

    stop_requested = False
    stopped_by_signal = False
    previous_handlers: list[tuple[signal.Signals, signal.Handlers]] = []

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal stop_requested, stopped_by_signal
        stop_requested = True
        stopped_by_signal = True
        logger.info("デーモン停止要求を受け付けました signal=%s", signum)

    if install_signal_handlers:
        for signum in _daemon_stop_signals():
            previous_handlers.append((signum, signal.getsignal(signum)))
            signal.signal(signum, request_stop)

    iterations = 0
    processed = 0
    no_issue = 0
    try:
        logger.info(
            "常駐デーモンを開始します interval_seconds=%s dry_run=%s max_iterations=%s",
            interval_seconds,
            dry_run,
            max_iterations,
        )
        while not stop_requested:
            if max_iterations is not None and iterations >= max_iterations:
                break

            iterations += 1
            processed_in_iteration = False
            for agent_context in agents:
                if stop_requested:
                    break
                result = run_once_func(
                    config=config,
                    agent=agent_context.profile,
                    redmine=agent_context.redmine,
                    task_executor=agent_context.task_executor,
                    dry_run=dry_run,
                    issue_id=None,
                )

                if result.status == "no_issue":
                    no_issue += 1
                    continue

                processed += 1
                processed_in_iteration = True
                logger.info(
                    "チケット処理を完了しました agent_id=%s issue_id=%s status=%s iteration=%s",
                    agent_context.profile.id,
                    result.issue_id,
                    result.status,
                    iterations,
                )

            if processed_in_iteration or stop_requested:
                continue
            logger.info(
                "全エージェントに処理対象がないため待機します interval_seconds=%s iteration=%s",
                interval_seconds,
                iterations,
            )
            if max_iterations is not None and iterations >= max_iterations:
                break
            try:
                sleeper(interval_seconds)
            except KeyboardInterrupt:
                stop_requested = True
                stopped_by_signal = True
                logger.info("デーモン待機中に停止要求を受け付けました")
    finally:
        for signum, handler in reversed(previous_handlers):
            signal.signal(signum, handler)

    logger.info(
        "常駐デーモンを終了します iterations=%s processed=%s no_issue=%s stopped_by_signal=%s",
        iterations,
        processed,
        no_issue,
        stopped_by_signal,
    )
    return DaemonResult(
        iterations=iterations,
        processed=processed,
        no_issue=no_issue,
        stopped_by_signal=stopped_by_signal,
    )


def _daemon_stop_signals() -> tuple[signal.Signals, ...]:
    signals = [signal.SIGINT]
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is not None:
        signals.append(sigterm)
    return tuple(signals)
