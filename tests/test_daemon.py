from __future__ import annotations

from typing import Any

from taskboard_agent.config import AgentProfileConfig, AppConfig
from taskboard_agent.daemon import AgentExecutionContext, run_daemon
from taskboard_agent.workflow import RunResult


AGENT = AgentProfileConfig(
    id="test-agent",
    redmine_user_id=42,
    redmine_api_key="redmine-key",
    llm_model="test-model",
    context_window_tokens=128000,
    llm_api_key="llm-key",
)

CONFIG = AppConfig(
    redmine_url="https://redmine.example.test",
    redmine_in_progress_status_id=2,
    redmine_review_status_id=10,
    linkace_url="https://linkace.example.test",
    linkace_api_key="linkace-key",
    linkace_summarized_list_id=10,
    agents=(AGENT,),
)

AGENTS = (
    AgentExecutionContext(
        profile=AGENT,
        redmine=object(),
        task_executor=object(),
    ),
)


class FakeRunOnce:
    def __init__(self, results: list[RunResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        config: AppConfig,
        agent: AgentProfileConfig,
        redmine: Any,
        task_executor: Any,
        dry_run: bool = False,
        issue_id: int | None = None,
    ) -> RunResult:
        self.calls.append(
            {
                "config": config,
                "agent": agent,
                "redmine": redmine,
                "task_executor": task_executor,
                "dry_run": dry_run,
                "issue_id": issue_id,
            }
        )
        if not self.results:
            return RunResult(status="no_issue", dry_run=dry_run)
        return self.results.pop(0)


def test_run_daemon_sleeps_only_when_no_issue_is_found() -> None:
    sleeps: list[float] = []
    run_once = FakeRunOnce(
        [
            RunResult(status="no_issue"),
            RunResult(status="no_issue"),
        ]
    )

    result = run_daemon(
        config=CONFIG,
        agents=AGENTS,
        interval_seconds=60,
        max_iterations=2,
        sleeper=sleeps.append,
        run_once_func=run_once,
        install_signal_handlers=False,
    )

    assert result.iterations == 2
    assert result.processed == 0
    assert result.no_issue == 2
    assert sleeps == [60]


def test_run_daemon_immediately_searches_again_after_processing_issue() -> None:
    sleeps: list[float] = []
    run_once = FakeRunOnce(
        [
            RunResult(status="processed", issue_id=101),
            RunResult(status="processed", issue_id=102),
            RunResult(status="no_issue"),
            RunResult(status="no_issue"),
        ]
    )

    result = run_daemon(
        config=CONFIG,
        agents=AGENTS,
        interval_seconds=60,
        max_iterations=4,
        sleeper=sleeps.append,
        run_once_func=run_once,
        install_signal_handlers=False,
    )

    assert result.iterations == 4
    assert result.processed == 2
    assert result.no_issue == 2
    assert sleeps == [60]


def test_run_daemon_continues_after_agent_failed_result() -> None:
    sleeps: list[float] = []
    run_once = FakeRunOnce(
        [
            RunResult(status="agent_failed", issue_id=101),
            RunResult(status="no_issue"),
            RunResult(status="no_issue"),
        ]
    )

    result = run_daemon(
        config=CONFIG,
        agents=AGENTS,
        interval_seconds=60,
        max_iterations=3,
        sleeper=sleeps.append,
        run_once_func=run_once,
        install_signal_handlers=False,
    )

    assert result.iterations == 3
    assert result.processed == 1
    assert result.no_issue == 2
    assert sleeps == [60]


def test_run_daemon_stops_at_max_iterations() -> None:
    sleeps: list[float] = []
    run_once = FakeRunOnce(
        [
            RunResult(status="processed", issue_id=101),
            RunResult(status="processed", issue_id=102),
        ]
    )

    result = run_daemon(
        config=CONFIG,
        agents=AGENTS,
        interval_seconds=60,
        max_iterations=2,
        sleeper=sleeps.append,
        run_once_func=run_once,
        install_signal_handlers=False,
    )

    assert result.iterations == 2
    assert len(run_once.calls) == 2
    assert sleeps == []


def test_run_daemon_passes_dry_run_and_uses_search_mode() -> None:
    run_once = FakeRunOnce([RunResult(status="no_issue", dry_run=True)])

    run_daemon(
        config=CONFIG,
        agents=AGENTS,
        dry_run=True,
        interval_seconds=60,
        max_iterations=1,
        sleeper=lambda _seconds: None,
        run_once_func=run_once,
        install_signal_handlers=False,
    )

    assert run_once.calls[0]["dry_run"] is True
    assert run_once.calls[0]["issue_id"] is None


def test_run_daemon_gives_each_agent_one_turn_per_iteration() -> None:
    second_profile = AgentProfileConfig(
        id="second-agent",
        redmine_user_id=43,
        redmine_api_key="second-redmine-key",
        llm_model="second-model",
        context_window_tokens=128000,
        llm_api_key="second-llm-key",
    )
    agents = (
        AGENTS[0],
        AgentExecutionContext(
            profile=second_profile,
            redmine=object(),
            task_executor=object(),
        ),
    )
    run_once = FakeRunOnce(
        [
            RunResult(status="processed", issue_id=101),
            RunResult(status="processed", issue_id=201),
            RunResult(status="processed", issue_id=102),
            RunResult(status="no_issue"),
        ]
    )
    sleeps: list[float] = []

    result = run_daemon(
        config=CONFIG,
        agents=agents,
        interval_seconds=60,
        max_iterations=2,
        sleeper=sleeps.append,
        run_once_func=run_once,
        install_signal_handlers=False,
    )

    assert [call["agent"].id for call in run_once.calls] == [
        "test-agent",
        "second-agent",
        "test-agent",
        "second-agent",
    ]
    assert result.processed == 3
    assert result.no_issue == 1
    assert sleeps == []
