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
