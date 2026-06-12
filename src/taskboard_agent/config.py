from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    redmine_url: str
    redmine_api_key: str
    redmine_ai_user_id: int
    openai_api_key: str
    openai_model: str


def load_config(env_file: str | Path = ".env") -> AppConfig:
    load_dotenv(dotenv_path=env_file, override=False)

    redmine_url = _required("REDMINE_URL").rstrip("/")
    redmine_api_key = _required("REDMINE_API_KEY")
    redmine_ai_user_id = _required_int("REDMINE_AI_USER_ID")
    openai_api_key = _required("OPENAI_API_KEY")
    openai_model = _required("OPENAI_MODEL")

    return AppConfig(
        redmine_url=redmine_url,
        redmine_api_key=redmine_api_key,
        redmine_ai_user_id=redmine_ai_user_id,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )


def _required(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigError(f"{name} is required")
    return value.strip()


def _required_int(name: str) -> int:
    value = _required(name)
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc

