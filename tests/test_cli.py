from __future__ import annotations

import pytest

from taskboard_agent.cli import build_parser


def test_run_once_parser_accepts_issue_id() -> None:
    args = build_parser().parse_args(["run-once", "--issue-id", "123"])

    assert args.issue_id == 123


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_run_once_parser_rejects_invalid_issue_id(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-once", "--issue-id", value])


def test_run_daemon_parser_uses_default_interval() -> None:
    args = build_parser().parse_args(["run-daemon"])

    assert args.interval_seconds == 60
    assert args.max_iterations is None


def test_run_daemon_parser_accepts_interval_and_max_iterations() -> None:
    args = build_parser().parse_args(
        ["run-daemon", "--interval-seconds", "30", "--max-iterations", "3"]
    )

    assert args.interval_seconds == 30
    assert args.max_iterations == 3


@pytest.mark.parametrize("option", ["--interval-seconds", "--max-iterations"])
@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_run_daemon_parser_rejects_invalid_positive_ints(
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-daemon", option, value])


def test_run_daemon_parser_requires_max_iterations_for_dry_run() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-daemon", "--dry-run"])


def test_run_daemon_parser_accepts_dry_run_with_max_iterations() -> None:
    args = build_parser().parse_args(
        ["run-daemon", "--dry-run", "--max-iterations", "1"]
    )

    assert args.dry_run is True
    assert args.max_iterations == 1
