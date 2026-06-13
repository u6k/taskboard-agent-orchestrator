from __future__ import annotations

import logging
from pathlib import Path

from taskboard_agent.logging_config import configure_logging, log_trace


def test_configure_logging_reads_logging_conf(tmp_path: Path) -> None:
    log_file = tmp_path / "agent.log"
    config_file = tmp_path / "logging.conf"
    config_file.write_text(
        "\n".join(
            [
                "[loggers]",
                "keys=root,test_logger",
                "",
                "[handlers]",
                "keys=file",
                "",
                "[formatters]",
                "keys=standard",
                "",
                "[logger_root]",
                "level=WARNING",
                "handlers=file",
                "",
                "[logger_test_logger]",
                "level=INFO",
                "handlers=file",
                "qualname=taskboard_agent.logging_config_test",
                "propagate=0",
                "",
                "[handler_file]",
                "class=FileHandler",
                "level=INFO",
                "formatter=standard",
                f"args=({str(log_file)!r}, 'w', 'utf-8')",
                "",
                "[formatter_standard]",
                "format=%(asctime)s [%(levelname)s] %(trace_id)s - %(message)s",
                "datefmt=%Y-%m-%d %H:%M:%S",
            ]
        ),
        encoding="utf-8",
    )

    configure_logging(config_file)
    logger = logging.getLogger("taskboard_agent.logging_config_test")

    with log_trace("issue#123"):
        logger.info("テストログ bookmark_id=%s", 99)

    logging.shutdown()

    output = log_file.read_text(encoding="utf-8")
    assert " [INFO] issue#123 - テストログ bookmark_id=99" in output


def test_log_trace_defaults_trace_id_to_dash(caplog) -> None:
    logger = logging.getLogger("taskboard_agent.logging_config_default_test")

    caplog.set_level(logging.INFO, logger=logger.name)
    with log_trace("issue#123"):
        logger.info("チケット処理中")
    logger.info("コンテキスト外")

    assert caplog.records[0].trace_id == "issue#123"
    assert caplog.records[1].trace_id == "-"
